#pragma once
#include <vector>
#include <map>

#if defined _WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>

	void* loadLibrary(std::string str){
		auto* l = LoadLibrary(std::wstring(str.begin(), str.end()).c_str());
		if (!l)
			std::cout << "ERROR: lib loading\n";
		return l;
	}
	void* loadFunction(void* library, std::string name){
		return GetProcAddress(library, name.c_str());
	}
	void closeLibrary(void* library){
		if (!FreeLibrary(library))
			std::cout << "ERROR: free lib\n";
	}
	std::string exportSpec = "__declspec(dllexport) ", libExp = ".lib", sharedLibExp = ".dll";
#else
	#include <dlfcn.h>
	void* loadLibrary(std::string str){
		auto* l = dlopen(str.c_str(), RTLD_LAZY);
		if (!l)
			fmt::print("ERROR: lib loading: {}\n", dlerror());
		return l;
	}
	void* loadFunction(void* library, std::string name){
		return dlsym(library, name.c_str());
	}
	void closeLibrary(void* library){
		if (dlclose(library))
			fmt::print("ERROR: lib freeing: {}\n", dlerror());
	}
	#define __cdecl __attribute__((__cdecl__))

	std::string exportSpec = "", libExp = ".o", sharedLibExp = ".so";
#endif


typedef float(__cdecl* cfwdfunc_t)();
typedef void(__cdecl* cbwdfunc_t)(float);

class DynamicLoader {
	std::string fileName = "_grad";
	std::string entireCode;
	std::map<std::string, cfwdfunc_t*> fwdfuncs;
	std::map<std::string, cbwdfunc_t*> bwdfuncs;
	void* library = nullptr;
public:
	DynamicLoader(std::vector<std::string> const& includeHeaders) {
		for(auto& h : includeHeaders)
			entireCode += fmt::format("#include <{}.h>\n", h);
		entireCode += "#define v(x) (*((float*)(x)))\n";
	}
	~DynamicLoader() {
		for (auto [k, v] : fwdfuncs)
			delete v;
		for (auto [k, v] : bwdfuncs)
			delete v;
		closeLibrary(library);
	}
	template<typename T>
	T* addFunction(std::string name, std::string code) {
		if(std::is_same_v<T,cfwdfunc_t>){
			entireCode += fmt::format("{}float {}() {{\n{}}}\n", exportSpec, name, code);
			auto fp = new cfwdfunc_t;
			fwdfuncs[name] = fp;
			return (T*)fp;
		}
		else if(std::is_same_v<T,cbwdfunc_t>){
			entireCode += fmt::format("{}void {}(float gradient) {{\n{}}}\n", exportSpec, name, code);
			auto fp = new cbwdfunc_t;
			bwdfuncs[name] = fp;
			return (T*)fp;
		}
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
			std::cout << "Mode is x64\n";
#else
			architectureFlag = "-m32";
			compiler = "tcc\\tcc.exe";
			std::cout << "Mode is x32\n";
#endif
			std::string args = "-O2 " + architectureFlag;

			AutoTimer at(g_timer, "compiler");
			system(fmt::format("{2} {0} -c -o {1}{3} {1}.c", args, fileName, compiler, libExp).c_str());
			//system(fmt::format("{2} {0} -c {1}.c", args, fileName, compiler).c_str());
			std::cout << "Created .lib\n";

			system(fmt::format("{2} {0} -shared -o {1}{4} {1}{3}", args, fileName, compiler, libExp, sharedLibExp).c_str());
			//system(fmt::format("{2} {0} -shared {1}.o", args, fileName, compiler).c_str());
			std::cout << "Created .dll\n";
		}

		std::string str = fmt::format("{}{}", fileName, sharedLibExp);
		library = loadLibrary("./"+str);

		for (auto& [name, fp] : fwdfuncs) {
			*fp = (cfwdfunc_t)loadFunction(library, name.c_str());
			if (!*fp) std::cout << "ERROR: loading func: "<<name<<"\n";
		}
		for (auto& [name, fp] : bwdfuncs) {
			*fp = (cbwdfunc_t)loadFunction(library, name.c_str());
			if (!*fp) std::cout << "ERROR: loading func: "<<name<<"\n";
		}
		std::cout << "Loaded .dll\n";
	}
};


