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
};

using implp_t = std::shared_ptr<impl>;

struct grad_fn {
	std::vector<implp_t> parents;
	virtual float operator()(int) = 0; // computes derivative wrt the i-th parent
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
};
struct subGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float operator()(int) override;
};
struct mulGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float operator()(int i) override;
};
struct divGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float operator()(int) override;
};
struct sqrtGrad : public grad1_fn {
	using grad1_fn::grad1_fn;
	float operator()(int) override;
};
struct powcGrad : public grad_fn {
	float exponent;
	powcGrad(implp_t l, float r) {
		parents = {l};
		exponent = r;
	}
	float operator()(int) override;
};
struct powGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float operator()(int) override;
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


	friend VE operator+(VE const& l, VE const& r) {
		return VE(l.value()+r.value(), 0, new addGrad(l.im, r.im));
	}
	friend VE operator-(VE const& l, VE const& r) {
		return VE(l.value()-r.value(), 0, new subGrad(l.im, r.im));
	}
	friend VE operator*(VE const& l, VE const& r) {
		return VE(l.value()*r.value(), 0, new mulGrad(l.im, r.im));
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

void impl::backward(float gradient) {
	grad += gradient;
	if (gradfn) { // if I am the result of an operation
		for (int i = 0;  const auto p : gradfn->parents) { // iteration over all the operands
			p->backward((*gradfn)(i) * gradient);
			++i;
		}
	}
}

void main() {


	VE a = 2, b = 5, c = 7;
	auto x = sqrt(pow(a*a+5*c,2*b-1));

	x.backward();
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



