import os, json

folder_a = "/scratch/ts1v23/workspace/reward_images/good"
folder_b = "/scratch/ts1v23/workspace/reward_images/bad"
output_json = "pairs.json"

# 读取图片列表
images_a = [os.path.join(folder_a, f) for f in os.listdir(folder_a) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
images_b = [os.path.join(folder_b, f) for f in os.listdir(folder_b) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

pairs = []
for pos in images_a:
    for neg in images_b:
        pairs.append({"chosen": pos, "rejected": neg})

# 保存
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(pairs, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(pairs)} pairs to {output_json}")
