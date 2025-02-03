import yaml

with open("./properties.yaml", "r") as properties_file:
    properties = yaml.safe_load(properties_file)

print(properties)
print(properties["SHOW_CLASS_NAME"])
if properties["SHOW_CLASS_NAME"]:
    print("hi")
