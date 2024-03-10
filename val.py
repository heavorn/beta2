from models.detect import YOLO

# w = "c2f-gs"

# # Load a model
# # model = YOLO('yolov8-p2.yaml') # load an official model
# model = YOLO(f'validation/detect_{w}_100.pt')  # load a custom model

w_list = ['msfnns']
n = 5

for w in w_list:
	map50 = []
	map75 = []
	mapall = []
	inference = []
	# model = YOLO(f'v2_{w}_300.pt')
	model = YOLO(f'weight/{w_list[0]}.pt')
	for _ in range(n):
		metrics = model.val(data='data.yaml', device='mps')
		map50.append(metrics.box.map50)
		map75.append(metrics.box.map75)
		mapall.append(metrics.box.map)
		inference.append(metrics.speed['inference'])

	print(f"Model: {w}")
	print(f"mAP(50): {round(sum(map50)/n, 4)}, mAP(75): {round(sum(map75)/n, 4)}, mAP(95): {round(sum(mapall)/n, 4)}, inference: {round(sum(inference)/n, 4)}")


# import pdb
# pdb.set_trace()

# Validate the model
# metrics = model.val(data='data.yaml', device='mps')  # no arguments needed, dataset and settings remembered
# metrics = model.val(data='data2.yaml', imgsz=640, device='mps', name=w)



# print('mAP(50-95): ', metrics.box.map)    # map50-95
# print('mAP(50): ', metrics.box.map50)  # map50
# print('mAP(75): ', metrics.box.map75)  # map75
# metrics.box.maps   # a list contains map50-95 of each category
