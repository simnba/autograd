#include <string>
#include <vector>
#include <iterator>

struct grad_fn;

struct impl {
	float value;
	float grad;
	grad_fn* gradfn;
	bool requiresGrad = true; 
	char varname = ' ';
	void update(); 
	void backward(float gradient = 1.f);
	void generateUpdate(std::stringstream& ss); 
	void generateBackward(std::stringstream& ss);
	int countElems() const;
	std::string printExpr() const;
	int getPrio() const;
};

using implp_t = std::shared_ptr<impl>;

struct grad_fn {
	std::vector<implp_t> parents;
	virtual float fwd() = 0;
	virtual float bwd(int i) = 0; // computes derivative wrt the i-th parent
	virtual void generateFwd(std::stringstream& ss, std::string const& old, std::string& comment) = 0;
	virtual void generateBwd(std::stringstream& ss, int i, std::string const& old, std::string& comment) = 0;
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
	float bwd(int) override {
		return 1;
	}
	void generateFwd(std::stringstream& ss, std::string const& old, std::string& comment) override {
		ss << fmt::format("v({0}) + v({1})",
						  (void*)(&parents[0]->value),
						  (void*)(&parents[1]->value));
		comment = "+";
	}
	void generateBwd(std::stringstream& ss, int, std::string const& old, std::string& comment) override {
		ss << old;
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
	float bwd(int i) override {
		return 1-2*i;
	}
	void generateFwd(std::stringstream& ss, std::string const& old, std::string& comment) override {
		ss << fmt::format("v({0}) - v({1})",
						  (void*)(&parents[0]->value),
						  (void*)(&parents[1]->value));
		comment = "-";
	}
	void generateBwd(std::stringstream& ss, int i, std::string const& old, std::string& comment) override {
		ss << (i==0?old:"-"+old);
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
	float bwd(int i) override {
		return parents[1-i]->value;
	}
	void generateFwd(std::stringstream& ss, std::string const& old, std::string& comment) override {
		ss << fmt::format("v({0}) * v({1})",
						  (void*)(&parents[0]->value),
						  (void*)(&parents[1]->value));
		comment = "*";
	}
	void generateBwd(std::stringstream& ss, int i, std::string const& old, std::string& comment) override {
		ss << fmt::format("{}*v({})", old, (void*)(&parents[1-i]->value));
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
	float bwd(int i) override {
		switch (i) {
		case 0:
			return 1.f/parents[1]->value;
		default:
			return -parents[0]->value/(parents[1]->value*parents[1]->value);
		}
	}
	void generateFwd(std::stringstream& ss, std::string const& old, std::string& comment) override {
		ss << fmt::format("v({0}) / v({1})",
						  (void*)(&parents[0]->value),
						  (void*)(&parents[1]->value));
		comment = "./.";
	}
	void generateBwd(std::stringstream& ss, int i, std::string const& old, std::string& comment) override {
		switch (i) {
		case 0:
			ss << fmt::format("{}/v({})", old, (void*)(&parents[1]->value));
			comment = "./";
			break;
		case 1:
			ss << fmt::format("-{0}*v({1})/(v({2})*v({2}))", old,
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
	float bwd(int) override {
		return 0.5f/sqrt(parents[0]->value);
	}
	void generateFwd(std::stringstream& ss, std::string const& old, std::string& comment) override {
		ss << fmt::format("sqrt(v({0}))",
						  (void*)(&parents[0]->value));
		comment = "sqrt";
	}
	void generateBwd(std::stringstream& ss, int i, std::string const& old, std::string& comment) override {
		ss << fmt::format("0.5f*{0}/sqrt(v({1}))", old,
						  (void*)(&parents[0]->value));
		comment = "sqrt";
	}
	std::string print(std::string l, std::string r) override { return "sqrt("+l+")"; }
	int getPrio() const { return 0; }
};
struct expGrad : public grad1_fn {
	using grad1_fn::grad1_fn;
	float fwd() override {
		return std::exp(parents[0]->value);
	}
	float bwd(int) override {
		return std::exp(parents[0]->value);
	}
	void generateFwd(std::stringstream& ss, std::string const& old, std::string& comment) override {
		ss << fmt::format("exp(v({0}))",
						  (void*)(&parents[0]->value));
		comment = "exp";
	}
	void generateBwd(std::stringstream& ss, int i, std::string const& old, std::string& comment) override {
		ss << fmt::format("{0}*exp(v({1}))", old,
						  (void*)(&parents[0]->value));
		comment = "exp";
	}
	std::string print(std::string l, std::string r) override { return "exp("+l+")"; }
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
	float bwd(int) override {
		return exponent * std::pow(parents[0]->value, exponent-1);
	}
	void generateFwd(std::stringstream& ss, std::string const& old, std::string& comment) override {
		if(exponent==2)
			ss << fmt::format("v({0})*v({0})", (void*)(&parents[0]->value));
		else
			ss << fmt::format("pow(v({0}),{1})", (void*)(&parents[0]->value), exponent);
		comment = ".^"+std::to_string(exponent);
	}
	void generateBwd(std::stringstream& ss, int, std::string const& old, std::string& comment) override {
		if(exponent == 2)
			ss << fmt::format("{1}*2*v({0})", (void*)(&parents[0]->value), old);
		else
			ss << fmt::format("{2}*{1}*pow(v({0}),{1}-1)", (void*)(&parents[0]->value), exponent,old);
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
	float bwd(int i) override {
		switch (i) {
		case 0:	return parents[1]->value * std::pow(parents[0]->value, parents[1]->value-1);
		case 1: return std::pow(parents[0]->value, parents[1]->value) * std::log(parents[0]->value);
		}
	}
	void generateFwd(std::stringstream& ss, std::string const& old, std::string& comment) override {
		ss << fmt::format("pow(v({0}),v({1}))", (void*)(&parents[0]->value), (void*)(&parents[1]->value));
		comment = ".^.";
	}
	void generateBwd(std::stringstream& ss, int i, std::string const& old, std::string& comment) override {
		switch (i) {
		case 0:	
			ss << fmt::format("{2}*{1}*pow(v({0}),v({1})-1)", (void*)(&parents[0]->value), (void*)(&parents[1]->value), old);
			comment = ".^";
			break;
		case 1: 
			ss << fmt::format("{2}*pow(v({0}),v({1})) * log(v({0}))", (void*)(&parents[0]->value), (void*)(&parents[1]->value), old);
			comment = "^."; 
			break;
		}
	}
	std::string print(std::string l, std::string r) override { return l + "^" + r; }
	int getPrio() const { return 3; }
};


// The class to use
class dual {
	std::shared_ptr<impl> im;
	cfwdfunc_t* fwdFunc = nullptr;
	cbwdfunc_t* bwdFunc = nullptr;
public:
	dual(float v = 0, bool requiresGrad = false) {
		im = std::make_shared<impl>(v, 0, nullptr, requiresGrad);
	}
	dual(grad_fn* gr) {
		// The result of an operation requires the gradient exactly if any of its operants requires it.
		bool requiresGrad = false;
		for (auto const& p : gr->parents) requiresGrad |= p->requiresGrad;
		im = std::make_shared<impl>(gr->fwd(), 0, gr, requiresGrad);
	}

	float& value() { return im->value; }
	const float& value() const { return im->value; }
	float& grad() { return im->grad; }
	const float& grad() const { return im->grad; }
	char& varName() {
		return im->varname;
	}
	bool& requiresGrad() {
		return im->requiresGrad;
	}

	void update() {
		im->update();
	}
	void backward(float gradient = 1.f) {
		im->backward(gradient);
	}
	void compile(DynamicLoader& dl) {
		AutoTimer at(g_timer, _FUNC_);
		{
			std::stringstream fwdCode;
			fwdCode << fmt::format("float value;\n");
			im->generateUpdate(fwdCode);
			fwdCode << fmt::format("return value;\n"); 
			fwdFunc = dl.addFunction<cfwdfunc_t>("forward", fwdCode.str());
		}
		{
			std::stringstream bwdCode;
			bwdCode << fmt::format("float* grad = 0;\n");
			im->generateBackward(bwdCode);
			bwdFunc = dl.addFunction<cbwdfunc_t>("backward", bwdCode.str());
		}
		dl.compileAndLoad();
	}
	void updateC() {
		im->value = (*fwdFunc)();
	}
	void backwardC(float gradient = 1.f) {
		(*bwdFunc)(gradient);
	}
	int getNumNodes() const {
		return im->countElems();
	}
	std::string exprToString() const {
		return im->printExpr();
	}
	

	dual& operator=(float v) {
		im->value = v;
		return *this;
	}

	friend dual operator+(dual const& l, dual const& r) {
		return dual(new addGrad(l.im, r.im));
	}
	friend dual operator-(dual const& l, dual const& r) {
		return dual(new subGrad(l.im, r.im));
	}
	friend dual operator*(dual const& l, dual const& r) {
		return dual(new mulGrad(l.im, r.im));
	}
	friend dual operator/(dual const& l, dual const& r) {
		return dual(new divGrad(l.im, r.im));
	}

	friend dual sqrt(dual const& l) {
		return dual(new sqrtGrad(l.im));
	}
	friend dual exp(dual const& l) {
		return dual(new expGrad(l.im));
	}
	friend dual pow(dual const& l, float r) {
		return dual(new powcGrad(l.im, r));
	}
	friend dual pow(dual const& l, dual const& r) {
		return dual(new powGrad(l.im, r.im));
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

// Implementations
void impl::update() {
	if (gradfn) {
		for (int i = 0; const auto& p : gradfn->parents)
			p->update();
		value = gradfn->fwd();
	}
}
void impl::backward(float gradient) {
	if (!requiresGrad)
		return;
	grad += gradient;
	if (gradfn)  // if I am the result of an operation
		for (int i = 0; const auto& p : gradfn->parents)  // iteration over all the operands
			p->backward(gradfn->bwd(i++) * gradient);
}
void impl::generateUpdate(std::stringstream& ss) {
	if (gradfn) {
		for (int i = 0; const auto& p : gradfn->parents)
			p->generateUpdate(ss);
		std::string comment;
		ss << fmt::format("value = v({}) = ", (void*)&value); 
		gradfn->generateFwd(ss, "unused", comment);
		ss << ";" << (comment.empty() ? "" : " //"+comment) << "\n"; 
	}
}
void impl::generateBackward(std::stringstream& ss) {
	if (!requiresGrad)
		return;
	ss << fmt::format("v({}) += gradient;\n", (void*)&grad);
	if (gradfn) {
		std::string old = fmt::format("g{}", (void*)this);
		ss << fmt::format("float {} = gradient;\n", old);
		for (int i = 0; const auto& p : gradfn->parents) {
			ss << fmt::format("gradient = ");
			std::string comment;
			gradfn->generateBwd(ss, i, old, comment);
			ss << ";" << (comment.empty() ? "" : " //"+comment) << "\n";
			p->generateBackward(ss);
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
