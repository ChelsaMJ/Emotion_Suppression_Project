from src.train import train_model

image_root = "data/CASME II/Cropped"
label_excel = "data/CASME II/CASME2-coding-20140508.xlsx"

train_model(
    image_root=image_root,
    label_excel=label_excel,
    num_classes=7,
    num_epochs=40
)


# -------- CROSS VALIDATION ----------
# cross_validate(
#     image_root=image_root,
#     label_excel=label_excel,
#     num_classes=7,
#     k=5,
#     epochs=5
# )