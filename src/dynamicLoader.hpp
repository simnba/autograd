#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <format>
#include <vector>
#include <map>

typedef float(__cdecl* cfwdfunc_t)();
typedef void(__cdecl* cbwdfunc_t)(float);

class DynamicLoader {
	std::string fileName = "_grad";
	std::string entireCode;
	std::map<std::string, cfwdfunc_t*> fwdfuncs;
	std::map<std::string, cbwdfunc_t*> bwdfuncs;
	HMODULE library = nullptr;
public:
	DynamicLoader(std::vector<std::string> const& includeHeaders) {
		for(auto& h : includeHeaders)
			entireCode += std::format("#include <{}.h>\n", h);
		entireCode += "#define v(x) (*((float*)(x)))\n";
	}
	~DynamicLoader() {
		for (auto [k, v] : fwdfuncs)
			delete v;
		for (auto [k, v] : bwdfuncs)
			delete v;
		if (!FreeLibrary(library))
			std::cout << "ERROR: free lib\n";
	}
	template<typename T>
	T* addFunction(std::string name, std::string code);

	template<>
	cfwdfunc_t* addFunction<cfwdfunc_t>(std::string name, std::string code) {
		entireCode += std::format("__declspec(dllexport) float {}() {{\n{}}}\n", name, code);
		auto fp = new cfwdfunc_t;
		fwdfuncs[name] = fp;
		return fp;
	}
	template<>
	cbwdfunc_t* addFunction<cbwdfunc_t>(std::string name, std::string code) {
		entireCode += std::format("__declspec(dllexport) void {}(float gradient) {{\n{}}}\n", name, code);
		auto fp = new cbwdfunc_t;
		bwdfuncs[name] = fp;
		return fp;
	}

	void compileAndLoad() {
		AutoTimer at(g_timer, _FUNC_);
		std::ofstream file(fileName+".c");
		file << entireCode;
		file.close();

		{
			std::string architectureFlag;
			std::string compiler;

#if defined(__x86_64__) or defined(_M_X64)
			architectureFlag = "-m64";
			compiler = "gcc";
#else
			architectureFlag = "-m32";
			compiler = "tcc\\tcc.exe";
#endif
			std::string args = "-O2 " + architectureFlag;

			AutoTimer at(g_timer, "compiler");
			system(std::format("{2} {0} -c -o {1}.lib {1}.c", args, fileName, compiler).c_str());
			std::cout << "Created .lib\n";
			system(std::format("{2} {0} -shared -o {1}.dll {1}.lib", args, fileName, compiler).c_str());
			std::cout << "Created .dll\n";
		}

		std::string str = std::format("{}.dll", fileName);
		library = LoadLibrary(std::wstring(str.begin(), str.end()).c_str());
		if (!library)
			std::cout << "ERROR: lib loading\n";

		for (auto& [name, fp] : fwdfuncs) {
			*fp = (cfwdfunc_t)GetProcAddress(library, name.c_str());
			if (!*fp) std::cout << "ERROR: loading func: "<<name<<"\n";
		}
		for (auto& [name, fp] : bwdfuncs) {
			*fp = (cbwdfunc_t)GetProcAddress(library, name.c_str());
			if (!*fp) std::cout << "ERROR: loading func: "<<name<<"\n";
		}
		std::cout << "Loaded .dll\n";
	}
};


