#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <memory>

struct grad_fn;
struct impl {
	float value;
	float grad;
	grad_fn* gradfn;
	void backward(float gradient = 1.f);
	int countElems() const;
	std::string printExpr() const;
	int getPrio() const;
};

using implp_t = std::shared_ptr<impl>;

struct grad_fn {
	std::vector<implp_t> parents;
	virtual float operator()(int) = 0; // computes derivative wrt the i-th parent
	virtual std::string print(std::string l, std::string r) = 0;
	virtual int getPrio() const = 0;
};
struct grad1_fn : grad_fn {
	grad1_fn(implp_t l) {
		parents = {l};
	}
}; 
struct grad2_fn : grad_fn {
	grad2_fn(implp_t l, implp_t r) {
		parents = {l,r};
	}
};
struct addGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float operator()(int) override;
	std::string print(std::string l, std::string r) override {return l+" + "+r;}
	int getPrio() const { return 1; }
};
struct subGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float operator()(int) override;
	std::string print(std::string l, std::string r) override { return l+" - "+r; }
	int getPrio() const { return 1; }
};
struct mulGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float operator()(int i) override;
	std::string print(std::string l, std::string r) override { return l+"*"+r; }
	int getPrio() const { return 2; }
};
struct divGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float operator()(int) override;
	std::string print(std::string l, std::string r) override { return l+"/"+r; }
	int getPrio() const { return 2; }
};
struct sqrtGrad : public grad1_fn {
	using grad1_fn::grad1_fn;
	float operator()(int) override;
	std::string print(std::string l, std::string r) override { return "sqrt("+l+")"; }
	int getPrio() const { return 0; }
};
struct powcGrad : public grad_fn {
	float exponent;
	powcGrad(implp_t l, float r) {
		parents = {l};
		exponent = r;
	}
	float operator()(int) override;
	std::string print(std::string l, std::string r) override { return l + "^" + r; }
	int getPrio() const { return 3; }
};
struct powGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float operator()(int) override;
	std::string print(std::string l, std::string r) override { return l + "^" + r; }
	int getPrio() const { return 3; }
};

class VE {
	
public:
	std::shared_ptr<impl> im;
	
	VE(float v = 0, float g = 0, grad_fn* gr = nullptr) {
		im = std::make_shared<impl>(v, g, gr);
	}
	auto& value(this auto&& self) {
		return self.im->value;
	}
	auto& grad(this auto&& self) {
		return self.im->grad;
	}
	void backward(float gradient = 1.f) {
		im->backward(gradient);
	}
	int countElems() const {
		return im->countElems();
	}
	std::string printExpr() const {
		return im->printExpr();
	}

	friend VE operator+(VE const& l, VE const& r) {
		return VE(l.value()+r.value(), 0, new addGrad(l.im, r.im));
	}
	friend VE operator-(VE const& l, VE const& r) {
		return VE(l.value()-r.value(), 0, new subGrad(l.im, r.im));
	}
	friend VE operator*(VE const& l, VE const& r) {
		return VE(l.value()*r.value(), 0, new mulGrad(l.im, r.im));
	}
	friend VE operator/(VE const& l, VE const& r) {
		return VE(l.value()/r.value(), 0, new divGrad(l.im, r.im));
	}
	friend VE sqrt(VE const& l) {
		return VE(std::sqrt(l.value()), 0, new sqrtGrad(l.im));
	}
	friend VE pow(VE const& l, float r) {
		return VE(std::pow(l.value(), r), 0, new powcGrad(l.im, r));
	}
	friend VE pow(VE const& l, VE const& r) {
		return VE(std::pow(l.value(), r.value()), 0, new powGrad(l.im, r.im));
	}
};

float addGrad::operator()(int i) {
	return 1;
}
float subGrad::operator()(int i) {
	return 1-2*i;
}
float mulGrad::operator()(int i) {
	return parents[1-i]->value;
}
float divGrad::operator()(int i) {
	switch (i) {
	case 0:
		return 1.f/parents[1]->value;
	case 1:
		return -parents[0]->value/(parents[1]->value*parents[1]->value);
	}
}
float sqrtGrad::operator()(int) {
	return 0.5f/sqrt(parents[0]->value);
}
float powcGrad::operator()(int) {
	return exponent * std::pow(parents[0]->value, exponent-1);
}
float powGrad::operator()(int i) {
	switch (i) {
	case 0:	return parents[1]->value * std::pow(parents[0]->value, parents[1]->value-1);
	case 1: return std::pow(parents[0]->value, parents[1]->value) * std::log(parents[0]->value);
	}
}

std::string tostr(float f) {
	std::ostringstream oss;
	oss << std::setprecision(3) << std::noshowpoint << f;
	return oss.str();
}
std::string bracket(std::string s) {
	return "("+s+")";
}

void impl::backward(float gradient) {
	grad += gradient;
	if (gradfn) { // if I am the result of an operation
		for (int i = 0;  const auto& p : gradfn->parents) { // iteration over all the operands
			p->backward((*gradfn)(i) * gradient);
			++i;
		}
	}
}
int impl::countElems() const {
	if (!gradfn)
		return 1;
	int i = 0;
	for (const auto& p : gradfn->parents)
		i += p->countElems();
	return i+1;
}
int impl::getPrio() const {
	if (gradfn)
		return abs(gradfn->getPrio());
	return 999;
}
std::string impl::printExpr() const {
	if (gradfn) {
		auto pl = gradfn->parents[0], pr = gradfn->parents[1];
		auto l = pl->printExpr(); if (pl->getPrio() <= getPrio()) l = bracket(l);
		auto r = pr->printExpr(); if (pr->getPrio() <= getPrio()) r = bracket(r);
		return gradfn->print(l,r);
	}
	else {
		return tostr(value);
	}
}

#include <random>
std::mt19937 gen(234);

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

void main() {

	VE a = 2, b = 5, c = 7;
	VE x = randomToken(8,8,{a,b,c});
	std::cout << "Value of x: " << x.value() << "\n";
	std::cout << "x = " << x.printExpr() << "\n";
	std::cout <<"Total elements in tree: " << x.countElems() << "\n";

	//auto x = sqrt(pow(a*a+5*c,2*b-1));

	auto t0 = std::chrono::high_resolution_clock::now();

	x.backward();

	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " ms\n";

	std::cout << a.grad() << "\n" << b.grad() << "\n" << c.grad() << std::endl;

	/*std::string dateTime = std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	std::ofstream file("test.c");

	std::cout << "Number is " << dateTime << std::endl;
	file << "\
		#include <stdio.h> \
		void main() { \
		printf(\"hallo " + dateTime + "\\n\");\
	}";
	file.close();

	system("tcc\\tcc.exe test.c");
	system("test.exe");*/

	std::cout << "The End";
	std::cin.get();
}



