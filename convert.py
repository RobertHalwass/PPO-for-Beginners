from network import PointNavNet
import coremltools as ct
import torch
import datetime

batch_size = 1

model = PointNavNet(3, 4, batch_size)
model.eval()
class_labels = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

example_visual_input = torch.rand((batch_size, 3, 224, 224))
example_pointgoal_input = torch.rand((batch_size,2))
example_last_action_input = torch.rand((batch_size,1))
traced_model = torch.jit.trace(model, (example_visual_input, example_pointgoal_input, example_last_action_input))

coreml_model = ct.converters.convert(
    traced_model,
    source="pytorch",
    inputs=[
        ct.TensorType(name="visual", shape=example_visual_input.shape),
        ct.TensorType(name="pointgoal", shape=example_pointgoal_input.shape),
        ct.TensorType(name="last_action", shape=example_last_action_input.shape)
    ], 
    classifier_config = ct.ClassifierConfig(class_labels)
)

coreml_model.input_description["visual"] = "Visual Input (Either RGB, Depth or RGB-Depth)"
coreml_model.input_description["pointgoal"] = "Polar-Coordinates Towards Point Goal"
coreml_model.input_description["last_action"] = "Last Action Taken By Agent"
coreml_model.output_description["classLabel"] = "Action To Take"

# Set model author name
coreml_model.author = 'Robert Halwaß'

# Set the license of the model
coreml_model.license = "MIT"

# Set a short description for the Xcode UI
coreml_model.short_description = "Visual Indoor Navigation"

# Set a version for the model
coreml_model.version = "1.0"

now = datetime.now() # current date and time
date_time = now.strftime("%m/%d/%Y-%H:%M:%S")

path = f"{date_time}.mlmodel"
coreml_model.save(path)
print("Saved model to: " + path)