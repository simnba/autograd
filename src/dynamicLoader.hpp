#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <format>
#include <map>

typedef void(__cdecl* cfunc_t)(float);
class DynamicLoader {
	std::string fileName = "_grad";
	std::string entireCode;
	std::map<std::string, cfunc_t*> funcs;
	HMODULE library = nullptr;
public:
	DynamicLoader() {
		entireCode = "#include <stdio.h>\n#define v(x) (*((float*)(x)))\n";
	}
	cfunc_t* addFunction(std::string name, std::string code) {
		entireCode += std::format("__declspec(dllexport) void {}(float gradient) {{\n{}\n}}\n", name, code);
		auto fp = new cfunc_t;
		funcs[name] = fp;
		return fp;
	}
	void compileAndLoad() {
		std::ofstream file(fileName+".c");
		file << entireCode;
		file.close();

		std::string args = "-m32";//-m64
		system(std::format("tcc\\tcc.exe {0} -c -o {1}.lib {1}.c", args, fileName).c_str());
		system(std::format("tcc\\tcc.exe {0} -shared -o {1}.dll {1}.lib", args, fileName).c_str());

		std::string str = std::format("{}.dll", fileName);
		library = LoadLibrary(std::wstring(str.begin(), str.end()).c_str());
		if (!library)
			std::cout << "ERROR: lib loading\n";

		for (auto [name, fp] : funcs) {
			*fp = (cfunc_t)GetProcAddress(library, name.c_str());
			if (!*fp)
				std::cout << "ERROR: loading func: "<<name<<"\n";
		}
	}
	~DynamicLoader() {
		if (!FreeLibrary(library))
			std::cout << "ERROR: free lib\n";
	}
};


