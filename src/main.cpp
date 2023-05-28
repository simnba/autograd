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
	//						 fmt::format("float* f = (float*){0}; *f *= 2;",(void*)&valu));
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
	//	std::cout << fmt::format("{}", a) << " -> " << *(float*)a << "\n";
	//}
}


template<bool COMPILED = false, bool VERBOSE = false>
void optimize(VE& loss, std::vector<VE>& vars, int niters, float step) {
	auto printVars = [&]() {
		std::cout << fmt::format("loss = {:8.4f}", loss.value());
		for (auto& v : vars)
			std::cout << fmt::format(", {} = {:8.4f}", v.varName(), v.value());
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
	
	printVars();
}

void perf() {
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

	std::cout << std::string(50, '-') << std::endl; // --------------------

	DynamicLoader dl({"math"});
	x.compile(dl);

	{
		AutoTimer at(g_timer, "Compiled");
		for (int i = 0; i < nreps; ++i) {
			resetVars();
			optimize<true, false>(x, vars, niters, step);
		}
	}
}

void linearRegression() {
	const int n = 50;
	float points[n][2];
	std::normal_distribution<> dist(0, 0.0001);
	for (int i = 0; i < n; ++i) {
		float x = (float)i/n; ;
		points[i][0] = x;
		points[i][1] = 1.2-2.3*x+x*x + dist(gen);
	}


	std::vector<float> initialValues = {1,1,1};
	std::vector<VE> vars(3);
	
	auto model = [&](float x)->VE {
		return vars[0] + vars[1]*x + vars[2]*x*x;
	};

	VE mse;
	for (int i = 0; i< n; ++i) {
		auto& [x, y] = points[i];
		mse = mse + pow(model(x)-y, 2);
	}
	mse = mse / n;

	auto resetVars = [&](bool compiled) {
		for (int i = 0; auto& v : vars) {
			v.value() = initialValues[i++];
		}
		if (compiled)
			mse.cForward();
		else
			mse.forward();
	};



	int nreps = 1;
	int niters = 5000;
	float step = 0.1;
	{
		AutoTimer at(g_timer, "Normal");
		for (int i = 0; i < nreps; ++i) {
			resetVars(false);
			optimize<false, false>(mse, vars, niters, step);
		}
	}

	std::cout << std::string(50, '-') << std::endl; // --------------------

	DynamicLoader dl({"math"});
	mse.compile(dl);

	{
		AutoTimer at(g_timer, "Compiled");
		for (int i = 0; i < nreps; ++i) {
			resetVars(true);
			optimize<true, false>(mse, vars, niters, step);
		}
	}
}

int main() {
	linearRegression();

	g_timer.print();
	std::cout << fmt::format("Speed-up: x{:.2}\n",
							 g_timer.getTotalSeconds("Normal")
							 /g_timer.getTotalSeconds("Compiled"));
	std::cin.get();
	return 0;
}
