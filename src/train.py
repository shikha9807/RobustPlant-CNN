import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import tensorflow as tf
from pathlib import Path
from src.config import ensure_dirs, set_global_seed
from src import config
from src.model import build_model, save_model_config, unfreeze_top_backbone
from src.dataset_loader import get_datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Train RobustPlant-CNN")
    parser.add_argument("--run-name", type=str, default="robustplant_b0", help="Name of the training run")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs-head", type=int, default=config.EPOCHS_HEAD, help="Epochs for head training")
    parser.add_argument("--epochs-finetune", type=int, default=config.EPOCHS_FINETUNE, help="Epochs for fine-tuning")
    parser.add_argument("--no-cbam", action="store_true", help="Disable CBAM attention modules")
    parser.add_argument("--weights", type=str, default="imagenet", help="Pretrained weights")
    return parser.parse_args()

def main():
    args = parse_args()
    set_global_seed()
    ensure_dirs()
    
    print(f"Starting training run: {args.run_name}")
    data = get_datasets(batch_size=args.batch_size, include_plantdoc=False)
    train_ds, val_ds = data["train_ds"], data["val_ds"]
    
    stages = [] if args.no_cbam else config.DEFAULT_CBAM_STAGES
    cfg = build_model.__globals__.get('ModelConfig')(
        backbone=config.BACKBONE,
        use_cbam=not args.no_cbam,
        cbam_stages=stages
    )
    
    model = build_model(cfg, weights=None if args.weights == "none" else "imagenet")
    print("Model built successfully!")
    
    checkpoint_path = config.MODELS_DIR / args.run_name / "checkpoint.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if a checkpoint exists to resume from
    if checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path)
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR_HEAD),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    ]
    
    # Phase 1: Train classification head
    print(f"\n--- Phase 1: Training Classification Head ({args.epochs_head} epochs) ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tuning
    print(f"\n--- Phase 2: Fine-Tuning Backbone ({args.epochs_finetune} epochs) ---")
    unfreeze_top_backbone(model, num_layers=config.UNFREEZE_LAYERS)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR_FINETUNE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        callbacks=callbacks
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
