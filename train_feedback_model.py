
import os
import time
from dotenv import load_dotenv
from adaption import Adaption

load_dotenv()

if not os.getenv("ADAPTION_API_KEY"):
    raise RuntimeError(
        "Set ADAPTION_API_KEY in your backend .env file."
    )

TRAINING_FILE = "BuildBattle_2026_Customer_Feedback_Dataset.csv"

if not os.path.exists(TRAINING_FILE):
    raise FileNotFoundError(
        f"Training file not found: {TRAINING_FILE}"
    )

client = Adaption()

# Upload the labeled training examples.

dataset = client.datasets.upload_file(
    TRAINING_FILE,
    processing_mode="raw",
    column_mapping={
        "prompt": "review",
        "completion": "category",
    },
)

dataset_id = dataset.dataset_id
print("Dataset ID:", dataset_id)

# Wait until Adaption has processed the dataset.
while True:
    status = client.datasets.get_status(dataset_id)

    if status.status == "failed":
        raise RuntimeError(
            f"Dataset ingestion failed: {status.error_data}"
        )

    if status.row_count is not None:
        print("Rows imported:", status.row_count)
        break

    time.sleep(5)

# Launch supervised model training.
run = client.autoscientist.create(
    dataset_id=dataset_id
)

print("Training run ID:", run.id)
print("Initial status:", run.status)

# Wait for training to finish.
run = client.autoscientist.wait_for_completion(run.id)

if run.status != "succeeded":
    raise RuntimeError(
        f"Training failed. Status: {run.status}; "
        f"Error: {run.error}"
    )

print("Training succeeded.")
print("Best win rate:", run.best_win_rate)

# Download the best checkpoint.
if not run.download_available:
    raise RuntimeError(
        "Training completed, but no checkpoint is available."
    )

with client.autoscientist.with_streaming_response.download(
    run.id
) as response:
    response.stream_to_file("best-checkpoint.tgz")

print("Saved checkpoint: best-checkpoint.tgz")
