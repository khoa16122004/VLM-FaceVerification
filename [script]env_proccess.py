import re
from pathlib import Path

def parse_package_list(file_path: str) -> list[tuple[str, str]]:
    packages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Package") or line.startswith("-"):
                continue
            parts = re.split(r"\s{2,}", line)
            if len(parts) >= 2:
                name, version = parts[0], parts[1]
                packages.append((name, version))
    return packages

def generate_conda_install(packages: list[tuple[str, str]]) -> str:
    return "conda install " + " ".join(f"{name}={version}" for name, version in packages)

def generate_environment_yml(packages: list[tuple[str, str]], env_name: str = "llava") -> str:
    lines = [
        f"name: {env_name}",
        "channels:",
        "  - conda-forge",
        "  - defaults",
        "dependencies:",
        "  - python=3.11"  # <-- chỉ định phiên bản Python tại đây
    ]
    lines += [f"  - {name}={version}" for name, version in packages]
    return "\n".join(lines)

def main():
    input_file = "llava_env.txt"  # bạn đổi tên file ở đây nếu cần
    packages = parse_package_list(input_file)

    conda_cmd = generate_conda_install(packages)
    print("📦 Conda install command:")
    print(conda_cmd)

    yml_content = generate_environment_yml(packages, env_name="llava")
    yml_file = Path("llava_environment.yml")
    yml_file.write_text(yml_content, encoding="utf-8")
    print(f"\n✅ Đã tạo file: {yml_file.resolve()}")

if __name__ == "__main__":
    main()
