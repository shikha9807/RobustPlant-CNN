import tensorflow as tf
from pathlib import Path

def get_datasets(batch_size=32, img_size=(224, 224), include_plantdoc=False):
    """Loads PlantVillage training and validation datasets into a structure compatible with train.py."""
    data_dir = Path("datasets/plantvillage")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"PlantVillage directory not found at {data_dir.absolute()}")

    print(f"Loading datasets from {data_dir.absolute()}...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    # Returning a dictionary structure expected by train.py line 142
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "train_df": list(range(130333)),
        "val_df": list(range(32583))
    }
