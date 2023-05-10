from inference_for_handler import semseg
from torchvision import io
import yaml

with open(r'configs/segformer_custom.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)


model = semseg(cfg)
image = io.read_image("data/basketball_train_data/images/validation/v4_00022200.jpg")
img = model.preprocess(image)
seg_map = model.model_forward(img)
seg_map = model.postprocess(image, seg_map, overlay=False)