pair_path = r"D:\VLM-FaceVerification\lfw\pairs.txt"
with open(pair_path, "r") as f:
    f.readline()  # bỏ dòng tiêu đề
    lines = [line.strip().split("\t") for line in f.readlines()]

num_same = 0
num_diff = 0
same_lines = []
diff_lines = []
exec_num = 500

for line in lines:
    if len(line) == 3 and num_same < exec_num:
        same_lines.append(line)
        num_same += 1
    elif len(line) == 4 and num_diff < exec_num:
        diff_lines.append(line)
        num_diff += 1
    if num_same == exec_num and num_diff == exec_num:
        break

output_path = "selected_pairs.txt"
with open(output_path, "w") as f:
    for line in same_lines:
        f.write("\t".join(line) + "\n")
    for line in diff_lines:
        f.write("\t".join(line) + "\n")

print(f"Đã ghi {len(same_lines)} dòng same và {len(diff_lines)} dòng diff vào '{output_path}'")
