#include <string>
#include <vector>

struct grad_fn;

struct impl {
	float value;
	float grad;
	grad_fn* gradfn;
	char varname = ' ';
	void update(); 
	void backward(float gradient = 1.f);
	void generateUpdate(std::stringstream& ss); 
	void generateBwd(std::stringstream& ss);
	int countElems() const;
	std::string printExpr() const;
	int getPrio() const;
};

using implp_t = std::shared_ptr<impl>;

struct grad_fn {
	std::vector<implp_t> parents;
	virtual float fwd() = 0;
	virtual float operator()(int i) = 0; // computes derivative wrt the i-th parent
	virtual void generateFwd(std::stringstream& ss, std::string& comment) = 0; 
	virtual void generateBwd(std::stringstream& ss, int i, std::string& comment) = 0;
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

// Implementations of all possible computations
struct addGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float fwd() override {
		return parents[0]->value + parents[1]->value;
	}
	float operator()(int) override {
		return 1;
	}
	void generateFwd(std::stringstream& ss, std::string& comment) override {
		ss << std::format("v({0}) + v({1})",
						  (void*)(&parents[0]->value),
						  (void*)(&parents[1]->value));
		comment = "+";
	}
	void generateBwd(std::stringstream& ss, int, std::string& comment) override {
		ss << "1";
		comment = "+";
	}
	std::string print(std::string l, std::string r) override { return l+" + "+r; }
	int getPrio() const { return 1; }
};
struct subGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float fwd() override {
		return parents[0]->value - parents[1]->value;
	}
	float operator()(int i) override {
		return 1-2*i;
	}
	void generateFwd(std::stringstream& ss, std::string& comment) override {
		ss << std::format("v({0}) - v({1})",
						  (void*)(&parents[0]->value),
						  (void*)(&parents[1]->value));
		comment = "-";
	}
	void generateBwd(std::stringstream& ss, int i, std::string& comment) override {
		ss << 1-2*i;
		comment = i==0 ? ".-" : "-.";
	}
	std::string print(std::string l, std::string r) override { return l+" - "+r; }
	int getPrio() const { return 1; }
};
struct mulGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float fwd() override {
		return parents[0]->value * parents[1]->value;
	}
	float operator()(int i) override {
		return parents[1-i]->value;
	}
	void generateFwd(std::stringstream& ss, std::string& comment) override {
		ss << std::format("v({0}) * v({1})",
						  (void*)(&parents[0]->value),
						  (void*)(&parents[1]->value));
		comment = "*";
	}
	void generateBwd(std::stringstream& ss, int i, std::string& comment) override {
		ss << std::format("v({})", (void*)(&parents[1-i]->value));
		comment = i==0 ? ".*" : "*.";
	}
	std::string print(std::string l, std::string r) override { return l+"*"+r; }
	int getPrio() const { return 2; }
};
struct divGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float fwd() override {
		return parents[0]->value / parents[1]->value;
	}
	float operator()(int i) override {
		switch (i) {
		case 0:
			return 1.f/parents[1]->value;
		case 1:
			return -parents[0]->value/(parents[1]->value*parents[1]->value);
		}
	}
	void generateFwd(std::stringstream& ss, std::string& comment) override {
		ss << std::format("v({0}) / v({1})",
						  (void*)(&parents[0]->value),
						  (void*)(&parents[1]->value));
		comment = "./.";
	}
	void generateBwd(std::stringstream& ss, int i, std::string& comment) override {
		switch (i) {
		case 0:
			ss << std::format("1.f/v({})", (void*)(&parents[1]->value));
			comment = "./";
			break;
		case 1:
			ss << std::format("-v({0})/(v({1})*v({1}))",
							  (void*)(&parents[0]->value), (void*)(&parents[1]->value));
			comment = "/.";
			break;
		}
	}
	std::string print(std::string l, std::string r) override { return l+"/"+r; }
	int getPrio() const { return 2; }
};
struct sqrtGrad : public grad1_fn {
	using grad1_fn::grad1_fn;
	float fwd() override {
		return std::sqrt(parents[0]->value);
	}
	float operator()(int) override {
		return 0.5f/sqrt(parents[0]->value);
	}
	void generateFwd(std::stringstream& ss, std::string& comment) override {
		ss << std::format("sqrt(v({0}))",
						  (void*)(&parents[0]->value));
		comment = "sqrt";
	}
	void generateBwd(std::stringstream& ss, int i, std::string& comment) override {
		ss << std::format("0.5f/sqrt(v({0}))",
						  (void*)(&parents[0]->value));
		comment = "sqrt";
	}
	std::string print(std::string l, std::string r) override { return "sqrt("+l+")"; }
	int getPrio() const { return 0; }
};
struct powcGrad : public grad_fn {
	float exponent;
	powcGrad(implp_t l, float r) {
		parents = {l};
		exponent = r;
	}
	float fwd() override {
		return std::pow(parents[0]->value,exponent);
	}
	float operator()(int) override {
		return exponent * std::pow(parents[0]->value, exponent-1);
	}
	void generateFwd(std::stringstream& ss, std::string& comment) override {
		ss << std::format("pow(v({0}),{1})", (void*)(&parents[0]->value), exponent);
		comment = ".^"+std::to_string(exponent);
	}
	void generateBwd(std::stringstream& ss, int, std::string& comment) override {
		ss << std::format("{1}*pow(v({0}),{1}-1)", (void*)(&parents[0]->value), exponent);
		comment = ".^"+std::to_string(exponent);
	}
	std::string print(std::string l, std::string r) override { return l + "^" + std::to_string(exponent); }
	int getPrio() const { return 3; }
};
struct powGrad : public grad2_fn {
	using grad2_fn::grad2_fn;
	float fwd() override {
		return std::pow(parents[0]->value, parents[1]->value);
	}
	float operator()(int i) override {
		switch (i) {
		case 0:	return parents[1]->value * std::pow(parents[0]->value, parents[1]->value-1);
		case 1: return std::pow(parents[0]->value, parents[1]->value) * std::log(parents[0]->value);
		}
	}
	void generateFwd(std::stringstream& ss, std::string& comment) override {
		ss << std::format("pow(v({0}),v({1}))", (void*)(&parents[0]->value), (void*)(&parents[1]->value));
		comment = ".^.";
	}
	void generateBwd(std::stringstream& ss, int i, std::string& comment) override {
		switch (i) {
		case 0:	
			ss << std::format("{1}*pow(v({0}),v({1})-1)", (void*)(&parents[0]->value), (void*)(&parents[1]->value));
			comment = ".^";
			break;
		case 1: 
			ss << std::format("pow(v({0}),v({1})) * log(v({0}))", (void*)(&parents[0]->value), (void*)(&parents[1]->value));
			comment = "^."; 
			break;
		}
	}
	std::string print(std::string l, std::string r) override { return l + "^" + r; }
	int getPrio() const { return 3; }
};


// The class to use
class VE {
	cfwdfunc_t* fwdFunc = nullptr; 
	cbwdfunc_t* bwdFunc = nullptr;
public:
	std::shared_ptr<impl> im;

	VE(float v = 0) {
		im = std::make_shared<impl>(v, 0, nullptr);
	}
	VE(grad_fn* gr) {
		im = std::make_shared<impl>(gr->fwd(), 0, gr);
	}
	VE& operator=(float v) {
		im->value = v;
		return *this;
	}
	auto& value(this auto&& self) {
		return self.im->value;
	}
	auto& grad(this auto&& self) {
		return self.im->grad;
	}
	void forward() {
		im->update();
	}
	void backward(float gradient = 1.f) {
		im->backward(gradient);
	}
	void compile(DynamicLoader& dl) {
		{
			std::stringstream fwdCode;
			fwdCode << std::format("float* value = 0;\n");
			im->generateUpdate(fwdCode);
			fwdCode << std::format("return *value;\n"); 
			fwdFunc = dl.addFunction<cfwdfunc_t>("forward", fwdCode.str());
		}
		{
			std::stringstream bwdCode;
			bwdCode << std::format("float* grad = 0;\n");
			im->generateBwd(bwdCode);
			bwdFunc = dl.addFunction<cbwdfunc_t>("backward", bwdCode.str());
		}
		dl.compileAndLoad();
	}
	void cForward() {
		im->value = (*fwdFunc)();
	}
	void cBackward(float gradient = 1.f) {
		(*bwdFunc)(gradient);
	}
	int countElems() const {
		return im->countElems();
	}
	std::string printExpr() const {
		return im->printExpr();
	}
	char& varName() const {
		return im->varname;
	}

	friend VE operator+(VE const& l, VE const& r) {
		return VE(new addGrad(l.im, r.im));
	}
	friend VE operator-(VE const& l, VE const& r) {
		return VE(new subGrad(l.im, r.im));
	}
	friend VE operator*(VE const& l, VE const& r) {
		return VE(new mulGrad(l.im, r.im));
	}
	friend VE operator/(VE const& l, VE const& r) {
		return VE(new divGrad(l.im, r.im));
	}
	friend VE sqrt(VE const& l) {
		return VE(new sqrtGrad(l.im));
	}
	friend VE pow(VE const& l, float r) {
		return VE(new powcGrad(l.im, r));
	}
	friend VE pow(VE const& l, VE const& r) {
		return VE(new powGrad(l.im, r.im));
	}
};


std::string tostr(float f) {
	std::ostringstream oss;
	oss << std::setprecision(3) << std::noshowpoint << f;
	return oss.str();
}
std::string bracket(std::string s) {
	return "("+s+")";
}
#include <set>
std::set<void*> addrs; // just for debugging


// Implementations
void impl::update() {
	if (gradfn) {
		for (int i = 0; const auto& p : gradfn->parents)
			p->update();
		value = gradfn->fwd();
	}
}
void impl::backward(float gradient) {
	grad += gradient;
	if (gradfn)  // if I am the result of an operation
		for (int i = 0; const auto& p : gradfn->parents)  // iteration over all the operands
			p->backward((*gradfn)(i++) * gradient);
}
void impl::generateUpdate(std::stringstream& ss) {
	if (gradfn) {
		for (int i = 0; const auto& p : gradfn->parents)
			p->generateUpdate(ss);
		std::string comment;
		ss << std::format("value = (float*){}; *value = ", (void*)&value);
		gradfn->generateFwd(ss, comment); 
		ss << ";" << (comment.empty() ? "" : " //"+comment) << "\n"; 
	}
}
void impl::generateBwd(std::stringstream& ss) {
	ss << std::format("grad = (float*){}; *grad += gradient;\n", (void*)&grad);
	if (gradfn) {
		std::string old = std::format("gradient_{}", (void*)this);
		ss << std::format("float {} = gradient;\n", old);
		for (int i = 0; const auto& p : gradfn->parents) {
			ss << std::format("gradient = {}*", old);
			std::string comment;
			gradfn->generateBwd(ss, i, comment);
			ss << ";" << (comment.empty() ? "" : " //"+comment) << "\n";
			p->generateBwd(ss);
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
	addrs.insert((void*)&value);
	if (gradfn) {
		auto pl = gradfn->parents[0];
		auto l = pl->printExpr(); if (pl->getPrio() <= getPrio()) l = bracket(l);
		std::string r;
		if (gradfn->parents.size()>1) {
			auto pr = gradfn->parents[1];
			r = pr->printExpr(); if (pr->getPrio() <= getPrio()) r = bracket(r);
		}
		return gradfn->print(l, r);
	}
	else {
		return varname == ' ' ? tostr(value) : std::string(1, varname);
	}
}
