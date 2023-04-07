from fastapi import FastAPI
from pyinpaint import Inpaint
from matplotlib import pyplot as plt

app = FastAPI()


@app.get("/")
async def root():
    inpaint = Inpaint("images/3.jpg", "images/3.jpg")
    inpainted_img = inpaint()
    show_images("images/3.jpg", "images/3.jpg", inpainted_img)
    return {"message": "Hello World"}


def show_images(org_img, mask, inpainted_img):
    org_img = plt.imread(org_img)
    org_img = (org_img - org_img.min()) / (org_img.max() - org_img.min())
    mask = plt.imread(mask)
    f = plt.figure(figsize=(20, 20))
    f.add_subplot(1, 3, 1)
    plt.imshow(org_img, cmap="gray")
    plt.axis("off")
    plt.title("ORIGINAL")
    f.add_subplot(1, 3, 2)
    plt.imshow((org_img.T * mask.T).T, cmap="gray")
    plt.axis("off")
    plt.title("MASKED")
    f.add_subplot(1, 3, 3)
    plt.imshow(inpainted_img, cmap="gray")
    plt.axis("off")
    plt.title("INPAINTED")
    # plt.savefig("out.jpg",bbox_inches="tight")
    plt.show()
