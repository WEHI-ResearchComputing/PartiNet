---
sidebar_position: 4
---

# Training Output Reference

This page describes the output generated during PartiNet training (both Step 1 and Step 2).

## Directory Structure

PartiNet creates a new experiment directory (`exp`, `exp2`, `exp3`, etc.) within your project folder for each training run. The experiment number automatically increments to avoid overwriting previous runs.

A **completed** training run will produce the following structure:
```
project_folder/
└── exp*/
    ├── cfg.yaml                      # Configuration parameters
    ├── hyp.yaml                      # Hyperparameters used
    ├── opt.yaml                      # Optimizer settings
    ├── LR.png                        # Learning rate schedule plot
    ├── results.png                   # Training/validation metrics over time
    ├── results.txt                   # Metrics in text format
    ├── confusion_matrix.png          # Model confusion matrix
    ├── F1_curve.png                  # F1 score curve
    ├── P_curve.png                   # Precision curve
    ├── R_curve.png                   # Recall curve
    ├── PR_curve.png                  # Precision-Recall curve
    ├── train_batch*.jpg              # Training batch visualizations
    ├── test_batch*_labels.jpg        # Validation ground truth
    ├── test_batch*_pred.jpg          # Validation predictions
    ├── events.out.tfevents.*         # TensorBoard logs
    └── weights/
        ├── best.pt                   # Best checkpoint (by validation metric)
        ├── last.pt                   # Most recent checkpoint
        └── epoch_*.pt                # Checkpoint for each epoch
```

## Understanding Training Visualizations

PartiNet generates several plots to help you evaluate model performance and training progress. You may find more in-depth guides of interpreting this data at https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc and https://docs.ultralytics.com/guides/yolo-performance-metrics/.

Below is a quick guide for intepreting PartiNet training outputs:

### Performance Curves

#### Learning Rate Plot (`LR.png`)
- Shows the learning rate schedule over training epochs
- Helps verify if learning rate warmup and decay are working as expected
- Look for: smooth warmup period followed by gradual decay

#### Results Plot (`results.png`)
- Displays training metrics over time:
  - Box loss: measures how well the model predicts particle bounding boxes
  - Objectness loss: indicates model's confidence in particle detection
  - Classification loss: not used in particle picking (always 0)
  - Precision and recall on validation set
- Look for: steadily decreasing losses and improving precision/recall

### Evaluation Metrics

#### Confusion Matrix (`confusion_matrix.png`)
- In particle picking, this is simplified since we only have one class
- Shows true positives, false positives, and false negatives
- Useful for understanding if model is:
  - Missing particles (false negatives)
  - Making spurious detections (false positives)

#### Precision-Recall Curves
- **PR Curve** (`PR_curve.png`): Shows the tradeoff between precision and recall
  - X-axis: Recall (percentage of actual particles detected)
  - Y-axis: Precision (percentage of detections that are actual particles)
  - Look for: curve that stays high (closer to 1.0) across different confidence thresholds

- **P Curve** (`P_curve.png`): Precision at different confidence thresholds
  - Higher curve indicates better precision across thresholds
  - Use to choose confidence threshold for inference

- **R Curve** (`R_curve.png`): Recall at different confidence thresholds
  - Shows how many particles are detected as threshold varies
  - Helps balance between missing particles and false positives

- **F1 Curve** (`F1_curve.png`): Harmonic mean of precision and recall
  - Single metric combining precision and recall
  - Peak indicates optimal confidence threshold for balanced performance

### Batch Visualizations

#### Training Batches (`train_batch*.jpg`)
- Shows model predictions on training data in a mosaic of image augmentations
- Blue boxes: Ground truth particle positions
- Look for:
  - Correct bounding boxes around particles even after image augmentation

#### Validation Results
- `test_batch*_labels.jpg`: Ground truth annotations
- `test_batch*_pred.jpg`: Model predictions on same images
- Compare these to assess model performance qualitatively
- Look for: 
  - Consistent detection of obvious particles
  - Few spurious detections in background areas
  - Good handling of challenging cases (overlapping particles, varying contrast)

## Monitoring Training Progress

**Real-time monitoring with TensorBoard:**
```bash
tensorboard --logdir /data/your_project_folder
```

Then open your browser to `http://localhost:6006` to view training metrics in real-time.

**Incomplete training runs** (due to timeout or errors) will have fewer outputs - only configuration files, learning rate plots, and training batch visualizations. Validation plots and test predictions only appear when training completes successfully.

## Resuming Interrupted Training

If training is interrupted due to timeout or out-of-memory errors, you can resume from the last checkpoint by pointing the `--weight` parameter to the `last.pt` file in your most recent experiment folder.

:::info Which Checkpoint to Use?
`best.pt` typically represents the checkpoint with the best validation performance, whereas `last.pt` contains weights for the last epoch, regardless of validation performance. **Overfitting of weights may occur with too many epochs of training**. This means `last.pt` may actually have worse performance than `best.pt`. It is important that you directly review validation performance during training to avoid this scenario
:::

:::tip Advanced Users
The YAML configuration files (`cfg.yaml`, `hyp.yaml`, `opt.yaml`) contain detailed information about the parameters and hyperparameters used during training. These can be useful for reproducing experiments or debugging training runs.
:::