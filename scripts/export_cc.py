import os

def xxd(file_path, var_name):
    with open(file_path, "rb") as f:
        data = f.read()
    
    out = f"// Automatically generated from {file_path}\n"
    out += f"#include <cstdint>\n\n"
    out += f"alignas(8) const unsigned char {var_name}[] = {{\n"
    
    for i, byte in enumerate(data):
        out += f"0x{byte:02x}, "
        if (i + 1) % 12 == 0:
            out += "\n"
            
    out += "\n};\n"
    out += f"const unsigned int {var_name}_len = {len(data)};\n"
    return out

def main():
    if os.path.exists("models/gesture_model_int8.tflite"):
        tflite_path = "models/gesture_model_int8.tflite"
    elif os.path.exists("models/gesture_model_int8_fallback.tflite"):
        tflite_path = "models/gesture_model_int8_fallback.tflite"
    else:
        tflite_path = "models/gesture_model_f32.tflite"
        
    print(f"Converting {tflite_path} to C array...")
    
    cc_content = xxd(tflite_path, "g_gesture_model_data")
    
    os.makedirs("firmware/include", exist_ok=True)
    with open("firmware/include/gesture_model_data.h", "w") as f:
        f.write(cc_content)
        
    print("Exported to firmware/include/gesture_model_data.h")

if __name__ == "__main__":
    main()
