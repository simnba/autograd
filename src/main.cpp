#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>

#include "timer.hpp"
#include "dynamicLoader.hpp"
#include "dual.hpp"


#include <random>
std::mt19937 gen(16); // Works for randomToken(8,8,{2,5,7})

dual randomToken(int minDepth, int maxDepth, std::vector<dual> const& vars) {
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
void optimize(dual& loss, std::vector<dual>& vars, int niters, float step) {
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
			loss.backwardC();
		else
			loss.backward();

		for(auto& v : vars)
			v.value() -= v.grad()*step;
		
		if (COMPILED)
			loss.updateC();
		else
			loss.update();

		if (VERBOSE)
			printVars();
	}
	
	printVars();
}

void perf() {
	std::vector<float> initialValues = {2,5,7};
	std::vector<dual> vars(3);
	for (int i = 0; auto& v : vars) {
		v.varName() = 'a'+(i++);
	}

	dual x = pow(randomToken(5, 5, vars) - 10, 2);
	std::cout << "x = " << x.exprToString() << "\n";

	auto resetVars = [&]() {
		for (int i = 0; auto& v : vars) {
			v.value() = initialValues[i++];
		}
		x.update();
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
	const int n = 7;
	float points[n][2];
	std::normal_distribution<> dist(0, 1);
	for (int i = 0; i < n; ++i) {
		float x = (float)i/n; ;
		points[i][0] = x;
		points[i][1] = 1.2-2.3*x+x*x + dist(gen);
	}

	std::vector<float> initialValues = {1,0.1,1,0.1};
	std::vector<dual> vars(4);
	for (auto& v : vars)
		v.requiresGrad() = true;

	auto model = [&](float x)->dual {
		dual b = vars[0] + exp(vars[1]) * dist(gen);
		dual m = vars[2] + exp(vars[3]) * dist(gen);
		return b + m*x;
	};

	int nSamples = 100;
	dual mse;
	for (int s = 0; s < nSamples; ++s){
		for (int i = 0; i < n; ++i) {
			auto& [x, y] = points[i];
			mse = mse + pow(model(x)-y, 2);
		}
	}
	mse = mse / (nSamples * n);

	auto resetVars = [&](bool compiled) {
		for (int i = 0; auto& v : vars) {
			v.value() = initialValues[i++];
		}
		if (compiled)
			mse.updateC();
		else
			mse.update();
	};



	int nreps = 1;
	int niters = 100;
	float step = 0.01;
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
