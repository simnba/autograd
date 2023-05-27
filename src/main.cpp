#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>

#include "timer.hpp"
#include "dynamicLoader.hpp"
#include "ve.hpp"


#include <random>
std::mt19937 gen(16); // Works for randomToken(8,8,{2,5,7})

VE randomToken(int minDepth, int maxDepth, std::vector<VE> const& vars) {
	std::uniform_int_distribution<> distrib((maxDepth<=0)*4, 3 + (minDepth <= 0)*5);
	std::uniform_int_distribution<> distrib2(0, vars.size()-1);
	int rand = distrib(gen);
	switch (rand) {
	case 0:
		return randomToken(minDepth-1, maxDepth-1, vars) + randomToken(minDepth-1, maxDepth-1, vars);
	case 1:
		return randomToken(minDepth-1, maxDepth-1, vars) - randomToken(minDepth-1, maxDepth-1, vars);
	case 2:
		return randomToken(minDepth-1, maxDepth-1, vars) * randomToken(minDepth-1, maxDepth-1, vars);
	case 3:
		return randomToken(minDepth-1, maxDepth-1, vars) / randomToken(minDepth-1, maxDepth-1, vars);
	default:
		return vars[distrib2(gen)];
	}
}

void testx64() {
	//float valu = 5;
	//DynamicLoader dl;
	//auto ff = dl.addFunction("change"+std::to_string((long)&valu),
	//						 std::format("float* f = (float*){0}; *f *= 2;",(void*)&valu));
	//dl.compileAndLoad();
	//(*ff)();

	//std::cout <<valu<<"\n";
	//std::cout << "The End";
	//std::cin.get(); 
	//
	//return;


	//// TEST addresses
	//x.printExpr();
	//for (auto& a : addrs) {
	//	std::cout << std::format("{}", a) << " -> " << *(float*)a << "\n";
	//}
}


template<bool COMPILED = false, bool VERBOSE = false>
void optimize(VE& loss, std::vector<VE> const& vars, int niters, float step) {
	auto printVars = [&]() {
		std::cout << std::format("loss={:8.4f}", loss.value());
		for (auto& v : vars)
			std::cout << std::format(", {} = {:8.4f}", v.varName(), v.value());
		std::cout << "\n";
	};

	
	for (int i = 0; i < niters; ++i) {
		for (auto& v : vars)
			v.grad() = 0;

		if (COMPILED)
			loss.cBackward();
		else
			loss.backward();

		for(auto& v : vars)
			v.value() -= v.grad()*step;
		
		if (COMPILED)
			loss.cForward();
		else
			loss.forward();

		if (VERBOSE)
			printVars();
	}
	
	//printVars();
	//std::cout << a.grad() << "\n" << b.grad() << "\n" << c.grad() << std::endl;
}


void main() {
	std::vector<float> initialValues = {2,5,7};
	std::vector<VE> vars(3);
	for (int i = 0; auto& v : vars) {
		v.varName() = 'a'+(i++);
	}

	VE x = pow(randomToken(5, 5, vars) - 10, 2);
	std::cout << "x = " << x.printExpr() << "\n";
	
	auto resetVars = [&]() {
		for (int i = 0; auto& v : vars) {
			v.value() = initialValues[i++];
		}
		x.forward();
	};

	int nreps = 10000;
	int niters = 100;
	float step = 0.0001;
	{
		AutoTimer at(g_timer, "Normal");
		for (int i = 0; i < nreps; ++i) {
			resetVars();
			optimize<false, false>(x, vars, niters, step);
		}
	}

	std::cout << std::string(50,'-') << std::endl; // --------------------

	DynamicLoader dl({"stdio","math"});
	x.compile(dl);

	{
		AutoTimer at(g_timer, "Compiled");
		for (int i = 0; i < nreps; ++i) {
			resetVars();
			optimize<true, false>(x, vars, niters, step);
		}
	}

	g_timer.print();
	std::cin.get();
}
