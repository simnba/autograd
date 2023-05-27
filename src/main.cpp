#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>

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
		return randomToken(minDepth-1, maxDepth-1, vars) * randomToken(minDepth-1, maxDepth-1, vars);
	default:
		return vars[distrib2(gen)];
	}
}


void main() {

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


	VE a = 2, b = 5, c = 7;
	a.varName() = 'a';
	b.varName() = 'b';
	c.varName() = 'c';
	//VE x = randomToken(8, 8, {a,b,c}); 
	VE x = randomToken(3, 3, {a,b,c});

	/*auto x = sqrt(pow(a*a+5*c,2*b-1));
	x.backward();*/
	float step = 0.001;
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000; ++i) {	
		a.grad() = b.grad() = c.grad() = 0;
		x.backward();
		a.value() -= a.grad()*step;
		b.value() -= b.grad()*step;
		c.value() -= c.grad()*step;
		std::cout << std::format("x={}, a={}, b={}, c={}\n", x.value(), a.value(), b.value(), c.value());
	}
	auto t1 = std::chrono::high_resolution_clock::now();

	std::cout << "x = " << x.value() << "\n";
	//std::cout << "x = " << x.printExpr() << "\n";
	std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " ms\n";
	std::cout << a.grad() << "\n" << b.grad() << "\n" << c.grad() << std::endl;
	x.printExpr();
	for (auto& a : addrs) {
		std::cout << std::format("{}", a) << " -> " << *(float*)a << "\n";
	}

	std::cout << std::string(30,'-') << std::endl; // --------------------


	DynamicLoader dl;
	x.compile(dl);

	t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000; ++i) {
		a.grad() = b.grad() = c.grad() = 0;

		x.compiledBackward();
	}
	t1 = std::chrono::high_resolution_clock::now();

	std::cout << "x = " << x.value() << "\n";
	//std::cout << "x = " << x.printExpr() << "\n";
	std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " ms\n";
	std::cout << a.grad() << "\n" << b.grad() << "\n" << c.grad() << std::endl;

		
	
	std::cout << "The End";
	std::cin.get();
}

