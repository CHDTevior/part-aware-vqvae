import os

def rename_images(folder, prefix=""):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort()  # 排序，保证顺序一致

    for i, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1].lower()  # 保留扩展名
        new_name = f"{prefix}{i:05d}{ext}"  # 00001.jpg
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)

    print(f"✅ Renamed {len(files)} files in {folder}")

# 用法
rename_images("/scratch/ts1v23/workspace/reward_images/good", prefix="good_")
rename_images("/scratch/ts1v23/workspace/reward_images/bad", prefix="bad_")

# /scratch/ts1v23/workspace/reward_images/good/good_00033.jpg