## Download DROID dataset

You can download the DROID dataset with the following command (after installing the `gsutil` google cloud CLI):
```bash
#gsutil -m cp -r gs://gresearch/robotics/droid <your_download_path>/droid/1.0.1
export DROID_DIR="~/data/DROID"
gsutil -m cp -r gs://gresearch/robotics/droid_100 $DROID_DIR
```


## Install SAM3

Update dependencies.

```bash
uv sync --group droid
```

Then run the preprocessing scripts. This will use SAM3 to mask the data and overlay affordance points.

```bash
# Fetches important metadata from DROID
uv run python examples/droid/preprocess_metadata.py --data_dir $DROID_DIR

# Peforms the preprocessing
uv run python examples/droid/preprocess_data.py --data_dir $DROID_DIR
```


## Potential data issues:
- The mask never gets occluded. If an object occuludes the gripper (e.g. a towel is folded over the top of the gripper), the gripper mask does not account for this and always gets drawn over the image. We could solve this by segmenting the object (e.g. with SAM3) and drawing the object over the gripper mask.
