"""
Get metadata about annotations and evaluate the annotators agreement.
"""

import os
import json 

import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from utils import (
    load_mask,
    process_circled_annotation,
    calculate_iou,
    generate_random_mask_like,
    load_image_np
)

# Global settings
plt.style.use('bmh') 
plt.rcParams['figure.autolayout'] = True

# Folder with original images
PATH_TO_ORIGINAL = os.path.join(
    os.path.dirname(__file__),
    "data",
    "annotations",
    "original"
)
# Folder with annotated iamges
PATH_TO_ANNOTATED = os.path.join(
    os.path.dirname(__file__),
    "data",
    "annotations",
    "annotated"
)
# metadata.json object
PATH_TO_METADATA = os.path.join(
    os.path.dirname(__file__),
    "data",
    "annotations",
    "metadata.json"
)
# Where to save plots
PATH_TO_PLOTS = os.path.join(
    os.path.dirname(__file__),
    "plots"
)


# Global var to store info
payload = {}

# Global var to cache
cache = {}
def load_with_cache(path_to_mask):
    if path_to_mask not in cache:
        mask = load_mask(
            os.path.join(PATH_TO_ANNOTATED, path_to_mask),
            target_size=(224,224)
        )
        mask = process_circled_annotation(mask)
        cache[path_to_mask] = mask
    return cache[path_to_mask]


def sanitize(df):
    """ 
    Sanitize the data fields
    """

    # Drop the test instances
    df = df[
        ~df.annotator_name.str.contains('test', case=False)
    ].copy()

    # Starting annotations
    payload["all_annotations"] = len(df)

    # Lowercase name
    df.annotator_name = df.annotator_name.str.lower()
    
    # Map professions
    df.annotator_profession = df.annotator_profession.str.lower()
    for pat,repl in {
        r".*studente.*":"Medical Student",
        r"specializzand.*":"Resident",
        r".*radiolog.*":"Radiologist",
        r".*medico.*":"Physician",
        r".*pneumolog.*":"Pneumologist"
    }.items():
        df.annotator_profession = df.annotator_profession.str.replace(
            pat=pat,
            repl=repl,
            regex=True
        )
    
    # Drop empty annotations
    idx = []
    nonzero = []
    for i in df.index:
        mask = load_with_cache(df.loc[i, "annotation_file"])
        
        sum = mask.sum()
        if sum:
            nonzero.append( sum / mask.size)
        else:
            idx.append(i)
    df = df.drop(index=idx)
    payload["nonzero_perc"] = np.mean(nonzero).item()

    return df

def render_thank_you(df):
    a = df.annotator_name.unique().tolist()
    names = []
    for i in a:
        if "test" in i.lower():
            continue
        splitted = i.split(" ")
        first_name = splitted[0].strip().capitalize()
        last_name = ""
        if len(splitted) > 1:
            last_name = splitted[1].strip().capitalize()[0]+"."
        name = first_name + " " + last_name
        name = name.strip()
        names.append(name)
    payload["thank_you"] = ", ".join(sorted(names))

def save_plots(df):
    """
    Make plots and save them.
    """
    # Annotations per image
    tmp = df.groupby("image_name").size()
    plt.figure(figsize=(6, 4))
    tmp.value_counts().sort_index().plot(kind='bar')
    plt.title('Distribution of Number of Annotations per Image')
    plt.xlabel('Number of Annotations')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(PATH_TO_PLOTS, "annotations_hist.png"))
    
    # Profession of annotators
    plt.figure(figsize=(6, 4))
    (
        df[["annotator_name", "annotator_profession"]]
        .drop_duplicates()
        .annotator_profession
        .value_counts()
        .plot(kind='bar')
    )
    plt.title('Distribution of Annotator Professions')
    plt.xlabel('Profession')
    plt.xticks(rotation=0)
    plt.ylabel('Number of Annotators')
    plt.savefig(os.path.join(PATH_TO_PLOTS, "demographics.png"))

def compute_cross_ious(df):
    """ 
    Measure inter-annotator agreement.
    """

    # Pair-wise IoUs
    results = []
    results_random = []
    for image_name, group in df.groupby("image_name"):

        # group = group[group.annotator_profession != "Medical Student"]
        # group = group[group.annotator_profession == "Medical Student"]
        files = sorted(group["annotation_file"].unique())  # sort for consistent ordering
        
        for i in range(len(files)):
            for j in range(i+1, len(files)):  # only unique pairs
                file1, file2 = files[i], files[j]
                mask1 = load_with_cache(file1)
                mask2 = load_with_cache(file2)
                mask_random = generate_random_mask_like(
                    mask2,
                    grid_size=5,
                    nonzero_perc=payload["nonzero_perc"]
                )
                if not mask1.sum() or not mask2.sum():
                    continue

                # Correct
                results.append({
                    "image_name": image_name,
                    "file1": file1,
                    "file2": file2,
                    "iou": calculate_iou(mask1, mask2)
                })

                # Random
                results_random.append({
                    "image_name": image_name,
                    "file1": file1,
                    "file2": file2,
                    "iou": calculate_iou(mask1, mask_random)
                })

    # Convert to dataset
    results = pd.DataFrame(results)
    results_random = pd.DataFrame(results_random)
    
    # Sanity check
    # Images with at least two annotations
    len_a = (df["image_name"].value_counts() >= 2).sum()
    # Number of images for which we computed IoU (should match)
    len_b = len(results.groupby("image_name").iou.mean())
    assert len_a == len_b
    
    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(results.groupby("image_name").iou.mean(), bins=25)
    plt.title("Average Pairwise IoU per Annotated Image")
    plt.ylabel("Number of Images")
    plt.xlabel("Average IoU")
    plt.savefig(os.path.join(PATH_TO_PLOTS, "average_ious"))

    # Plot random image, with annotation and random mask
    i = np.random.choice(df.index, size=(1,)).item()
    plt.figure(figsize=(6, 4))

    plt.subplot(131)
    img = load_image_np(
        os.path.join(PATH_TO_ORIGINAL, df.loc[i, "image_name"])
    )
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(132)
    plt.title("True")
    mask = load_with_cache(
            os.path.join(
                PATH_TO_ANNOTATED,
                df.loc[i, "annotation_file"]
            )
    )
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    
    plt.subplot(133)
    plt.title("Random")
    plt.imshow(generate_random_mask_like(
                    mask,
                    grid_size=5,
                    nonzero_perc=payload["nonzero_perc"]
                ), cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(PATH_TO_PLOTS, "random_mask.png"))
    
    # Perform statistical test to get p-value
    # H_0: iou is not greater than random
    # --> p < 0.05 --> iou is greater than random iou
    t_stat, p_value = stats.mannwhitneyu(
        results.iou,
        results_random.iou,
        alternative='greater'
    )
    
    # Store overall mean and p-value
    payload["avg_pairwise_iou"] = results.iou.mean().item()
    payload["avg_pairwise_iou_random"] = results_random.iou.mean().item()
    payload["iou_vs_random_p_value"] = p_value.item() 

def main():

    assert os.path.exists(PATH_TO_ORIGINAL)
    assert os.path.exists(PATH_TO_ANNOTATED)

    # Load metadata
    df = pd.read_json(PATH_TO_METADATA)
    
    # Sanitize
    df = sanitize(df)
    
    # Save thank you mex to payload
    render_thank_you(df)
    
    # Get metadata
    payload["valid_annotations"] = len(df)
    payload["unique_annotators"] = len(df.annotator_name.unique())
    payload["avg_annotations_img"] = df.groupby("image_name").size().mean().item()
    payload["unique_annotated_img"] = len(df.image_name.unique())
    
    # Frequency of annotations
    save_plots(df)
    
    # Measure consensus
    compute_cross_ious(df)

    # Save & display metadata measured
    path = os.path.join(PATH_TO_PLOTS, "annotations_info.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Saved payload to {path}:")
    print(payload)

    # Save the final dataframe
    df.to_csv(
        os.path.join(
            os.path.dirname(PATH_TO_ANNOTATED),
            "clean_metadata.csv")
    )

if __name__ == "__main__":
    main()