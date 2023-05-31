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

/* max. rel. error = 3.55959567e-2 on [-87.33654, 88.72283] */
__m128 FastExpSse(__m128 x)
{
	__m128 a = _mm_set1_ps(12102203.0f); /* (1 << 23) / log(2) */
	__m128i b = _mm_set1_epi32(127 * (1 << 23) - 298765);
	__m128i t = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(a, x)), b);
	return _mm_castsi128_ps(t);
}

template<bool COMPILED = false>
void optimize(dual& loss, std::vector<dual>& vars, int niters, float step, std::function<void()> const& printVars = nullptr) {
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

		if (printVars)
			printVars();
	}
}

//void perf() {
//	std::vector<float> initialValues = {2,5,7};
//	std::vector<dual> vars(3);
//	for (int i = 0; auto& v : vars) {
//		v.setVarName(""+('a'+(i++)));
//	}
//
//	dual x = pow(randomToken(5, 5, vars) - 10, 2);
//	std::cout << "x = " << x.exprToString() << "\n";
//
//	auto resetVars = [&]() {
//		for (int i = 0; auto& v : vars) {
//			v.value() = initialValues[i++];
//		}
//		x.update();
//	};
//
//	int nreps = 10000;
//	int niters = 100;
//	float step = 0.0001;
//	{
//		AutoTimer at(g_timer, "Normal");
//		for (int i = 0; i < nreps; ++i) {
//			resetVars();
//			optimize<false, false>(x, vars, niters, step);
//		}
//	}
//
//	std::cout << std::string(50, '-') << std::endl; // --------------------
//
//	DynamicLoader dl({"math"});
//	x.compile(dl);
//
//	{
//		AutoTimer at(g_timer, "Compiled");
//		for (int i = 0; i < nreps; ++i) {
//			resetVars();
//			optimize<true, false>(x, vars, niters, step);
//		}
//	}
//}

void linearRegression() {
	
	const int nPoints = 7;
	int nSamples = 2;
	int nReps = 1;
	int nIters = 1000;
	float step = 0.05;

	// Generate points
	float points[nPoints][2];
	std::normal_distribution<> dist(0, 1);
	for (int i = 0; i < nPoints; ++i) {
		float x = (float)i/nPoints; ;
		points[i][0] = x;
		points[i][1] = 1.2 - 2.3*x + x*x*0 + 0.1*dist(gen);
	}

	// Save points to file
	std::ofstream pointFile("points.txt");
	for (auto& p : points)
		pointFile << p[0] << " " << p[1] << "\n";
	pointFile.close();
	
	struct Model {
		std::vector<float> initialValues = {1,0.1,1,0.1};
		std::vector<dual> vars;
		dual b, m;
		Model() {
			vars.resize(4);
			for (auto& v : vars)
				v.setRequiresGrad(true);
			vars[0].setVarName("bmu");
			vars[1].setVarName("bsg");
			vars[2].setVarName("mmu");
			vars[3].setVarName("msg");
			reset();
			sample();
		}
		void sample() {
			static std::normal_distribution<> dist(0, 1);
			b = vars[0] + exp(vars[1]) * dist(gen);
			m = vars[2] + exp(vars[3]) * dist(gen);
		}
		dual operator()(float x)  {
			return b + m*x;
		};
		void reset() {
			for (int i = 0; auto& v : vars) {
				v.value() = initialValues[i++];
			}
		};
	} model;
	

	dual mse;
	for (int s = 0; s < nSamples; ++s){
		model.sample();
		for (int i = 0; i < nPoints; ++i) {
			auto& [x, y] = points[i];
			mse = mse + pow(model(x)-y, 2);
		}
	}
	mse = mse / (nSamples * nPoints);

	auto printVars = [&]() {
		std::cout << fmt::format("loss = {:8.4f}", mse.value());
		for (auto& v : model.vars)
			std::cout << fmt::format(", {} = {:8.4f}", v.getVarName(), v.value());
		std::cout << "\n";
	};

	std::cout << mse.getExprString() << "\n";
	auto counter = mse.getNumNodes();
	fmt::print("Leaf count: {}, num constants: {}, num req. gradient: {}, num nograd: {}\n",
			   counter.nNodes, counter.nConstants, counter.nReqGrad, counter.nNodes-counter.nConstants-counter.nReqGrad);

	{
		AutoTimer at(g_timer, "Normal");
		for (int i = 0; i < nReps; ++i) {
			model.reset();
			mse.update();
			optimize<false>(mse, model.vars, nIters, step, printVars);
		}
	}
	printVars();

	std::cout << std::string(50, '-') << std::endl; // --------------------

	/*DynamicLoader dl({"math"});
	mse.compile(dl);

	{
		AutoTimer at(g_timer, "Compiled");
		for (int i = 0; i < nReps; ++i) {
			model.reset();
			mse.updateC();
			optimize<true>(mse, model.vars, nIters, step);
		}
	}
	printVars();*/

	// Save result parameters to file
	std::ofstream paramFile("params.txt");
	for (auto& v : model.vars)
		paramFile << v.value() << "\n";
	paramFile.close();
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
