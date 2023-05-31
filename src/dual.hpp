#include <string>
#include <vector>
#include <iterator>
#include <set>

std::string tostr(float f) {
	std::ostringstream oss;
	oss << std::setprecision(3) << std::noshowpoint << f;
	return oss.str();
}
std::string toHexFloatStr(float f) {
	std::ostringstream oss;
	oss << std::hexfloat << f;
	return oss.str();
}
std::string bracket(std::string s) {
	return "("+s+")";
}

template<class T> concept enumeration = std::is_enum_v<T>;
template<enumeration T> inline T operator~ (T a) { return (T)~(int)a; }
template<enumeration T> inline T operator| (T a, T b) { return (T)((int)a | (int)b); }
template<enumeration T> inline T operator& (T a, T b) { return (T)((int)a & (int)b); }
template<enumeration T> inline T operator^ (T a, T b) { return (T)((int)a ^ (int)b); }
template<enumeration T> inline T& operator|= (T& a, T b) { return (T&)((int&)a |= (int)b); }
template<enumeration T> inline T& operator&= (T& a, T b) { return (T&)((int&)a &= (int)b); }
template<enumeration T> inline T& operator^= (T& a, T b) { return (T&)((int&)a ^= (int)b); }

struct nodeCountInfo {
	int nNodes = 0, nConstants = 0, nReqGrad = 0;
};


struct operation;

struct expr {
	static inline std::map<const expr*, std::string> namedict;
	float value;
	float grad;
	operation* op;
	enum EFlags {
		boring = 0,
		requiresGrad = 1,
		constant = 2
	}flags = boring;
	expr(float v, float g, operation* o, bool rg, bool c) :value{v}, grad{g}, op{o}, flags{(EFlags)(rg*requiresGrad | c*constant)}{}
	void update(); 
	void backward(float gradient = 1.f);
	void generateUpdate(std::stringstream& ss, std::set<expr const*>& visited);
	void generateBackward(std::stringstream& ss, std::set<expr const*>& visited);
	void countElems(nodeCountInfo& counter) const;
	std::string printExpr() const;
	int getPrio() const;
	
	std::string getVarName() const {
		if (expr::namedict.contains(this))
			return expr::namedict.at(this);
		else
			return "";
	}
};

using exprp_t = std::shared_ptr<expr>;

struct operation {
	std::vector<exprp_t> parents;
	virtual float fwd() = 0;
	virtual float bwd(int i) = 0; // computes derivative wrt the i-th parent
	virtual void generateFwd(std::stringstream& ss, std::string const& old, std::string& comment) = 0;
	virtual void generateBwd(std::stringstream& ss, int i, std::string const& old, expr const& result, std::string& comment) = 0;
	virtual std::string print(std::string l, std::string r) = 0;
	virtual int getPrio() const = 0;

	std::string resolveValue(exprp_t const& p) {
		if (p->flags & expr::constant)
			return toHexFloatStr(p->value);
		else
			return fmt::format("v({})",(void*)(&p->value));
	}
};
struct grad1_fn : operation {
	grad1_fn(exprp_t l) {
		parents = {l};
	}
};
struct grad2_fn : operation {
	grad2_fn(exprp_t l, exprp_t r) {
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
		ss << fmt::format("{} + {}",
						  resolveValue(parents[0]),
						  resolveValue(parents[1]));
		comment = "+";
	}
	void generateBwd(std::stringstream& ss, int, std::string const& old, expr const& result, std::string& comment) override {
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
		ss << fmt::format("{} - {}",
						  resolveValue(parents[0]),
						  resolveValue(parents[1]));
		comment = "-";
	}
	void generateBwd(std::stringstream& ss, int i, std::string const& old, expr const& result, std::string& comment) override {
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
		ss << fmt::format("{} * {}",
						  resolveValue(parents[0]),
						  resolveValue(parents[1]));
		comment = "*";
	}
	void generateBwd(std::stringstream& ss, int i, std::string const& old, expr const& result, std::string& comment) override {
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
		ss << fmt::format("{} / {}",
						  resolveValue(parents[0]),
						  resolveValue(parents[1]));
		comment = "./.";
	}
	void generateBwd(std::stringstream& ss, int i, std::string const& old, expr const& result, std::string& comment) override {
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
	void generateBwd(std::stringstream& ss, int i, std::string const& old, expr const& result, std::string& comment) override {
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
	void generateBwd(std::stringstream& ss, int i, std::string const& old, expr const& result, std::string& comment) override {
		ss << fmt::format("{0}*v({1})", old, (void*)(&result.value));
		comment = "exp";
	}
	std::string print(std::string l, std::string r) override { return "Exp["+l+"]"; }
	int getPrio() const { return 0; }
};
struct powcGrad : public operation {
	float exponent;
	powcGrad(exprp_t l, float r) {
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
	void generateBwd(std::stringstream& ss, int, std::string const& old, expr const& result, std::string& comment) override {
		if(exponent == 2)
			ss << fmt::format("{1}*2*v({0})", (void*)(&parents[0]->value), old);
		else
			ss << fmt::format("{2}*{1}*pow(v({0}),{1}-1)", (void*)(&parents[0]->value), exponent,old);
		comment = ".^"+std::to_string(exponent);
	}
	std::string print(std::string l, std::string r) override { return l + "^" + tostr(exponent); }
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
	void generateBwd(std::stringstream& ss, int i, std::string const& old, expr const& result, std::string& comment) override {
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
	exprp_t ex;
	cfwdfunc_t* fwdFunc = nullptr;
	cbwdfunc_t* bwdFunc = nullptr;
public:
	dual(float v = 0, bool requiresGrad = false) {
		ex = std::make_shared<expr>(v, 0, nullptr, requiresGrad, !requiresGrad);
	}
	dual(operation* op) {
		// The result of an operation requires the gradient exactly if any of its operants requires it.
		bool requiresGrad = false;
		for (auto const& p : op->parents) requiresGrad |= p->requiresGrad;
		ex = std::make_shared<expr>(op->fwd(), 0, op, requiresGrad, false);
	}

	float& value() { return ex->value; }
	const float& value() const { return ex->value; }
	float& grad() { return ex->grad; }
	const float& grad() const { return ex->grad; }
	std::string getVarName() const {
		return ex->getVarName();
	}
	void setVarName(std::string const& name) {
		expr::namedict.insert(std::make_pair(ex.get(), name));
	}
	
	bool getRequiresGrad() const {
		return ex->flags & expr::requiresGrad;
	}
	void setRequiresGrad(bool b) {
		ex->flags |= expr::requiresGrad;
		if (b)
			ex->flags &= ~expr::constant;
	}

	void update() {
		AutoTimer at(g_timer, _FUNC_);
		ex->update();
	}
	void backward(float gradient = 1.f) {
		AutoTimer at(g_timer, _FUNC_);
		ex->backward(gradient);
	}
	void compile(DynamicLoader& dl) {
		AutoTimer at(g_timer, _FUNC_);
		{
			std::set<expr const*> visited;
			std::stringstream fwdCode;
			fwdCode << fmt::format("float v;\n");
			ex->generateUpdate(fwdCode, visited);
			fwdCode << fmt::format("return v;\n"); 
			fwdFunc = dl.addFunction<cfwdfunc_t>("forward", fwdCode.str());
		}
		{
			std::set<expr const*> visited;
			std::stringstream bwdCode;
			ex->generateBackward(bwdCode, visited);
			bwdFunc = dl.addFunction<cbwdfunc_t>("backward", bwdCode.str());
		}
		dl.compileAndLoad();
	}
	void updateC() {
		AutoTimer at(g_timer, _FUNC_);
		ex->value = (*fwdFunc)();
	}
	void backwardC(float gradient = 1.f) {
		AutoTimer at(g_timer, _FUNC_);
		(*bwdFunc)(gradient);
	}


	nodeCountInfo getNumNodes() const {
		nodeCountInfo counter;
		ex->countElems(counter);
		return counter;
	}
	std::string getExprString() const {
		return ex->printExpr();
	}
	

	dual& operator=(float v) {
		ex->value = v;
		return *this;
	}

	friend dual operator+(dual const& l, dual const& r) {
		return dual(new addGrad(l.ex, r.ex));
	}
	friend dual operator-(dual const& l, dual const& r) {
		return dual(new subGrad(l.ex, r.ex));
	}
	friend dual operator*(dual const& l, dual const& r) {
		return dual(new mulGrad(l.ex, r.ex));
	}
	friend dual operator/(dual const& l, dual const& r) {
		return dual(new divGrad(l.ex, r.ex));
	}

	friend dual sqrt(dual const& l) {
		return dual(new sqrtGrad(l.ex));
	}
	friend dual exp(dual const& l) {
		return dual(new expGrad(l.ex));
	}
	friend dual pow(dual const& l, float r) {
		return dual(new powcGrad(l.ex, r));
	}
	friend dual pow(dual const& l, dual const& r) {
		return dual(new powGrad(l.ex, r.ex));
	}
};


// Implementations
void expr::update() {
	if (op) {
		for (int i = 0; const auto& p : op->parents)
			p->update();
		value = op->fwd();
	}
}
void expr::backward(float gradient) {
	grad += gradient;
	if (op)  // if I am the result of an operation
		for (int i = 0; const auto& p : op->parents) { // iteration over all the operands
			if (p->requiresGrad)
				p->backward(op->bwd(i) * gradient);
			++i;
		}
}
void expr::generateUpdate(std::stringstream& ss, std::set<expr const*>& visited) {
	if (op) {
		bool vis = visited.contains(this);
		visited.insert(this);
		if (vis)
			return; // The graph has a circle and we have computed this value already
		for (int i = 0; const auto& p : op->parents)
			p->generateUpdate(ss, visited);
		std::string comment;
		ss << fmt::format("v=v({}) = ", (void*)&value); 
		op->generateFwd(ss, "unused", comment);
		ss << ";" << (comment.empty() ? "" : " //"+comment) << "\n"; 
	}
}
void expr::generateBackward(std::stringstream& ss, std::set<expr const*>& visited) {
	ss << fmt::format("v({}) += gradient;\n", (void*)&grad);
	if (op) {
		bool vis = visited.contains(this);
		visited.insert(this);
		std::string old = fmt::format("g{}", (void*)this);
		if (!vis)
			ss << "float ";
		ss << fmt::format("{} = gradient; \n", old);
		for (int i = 0; const auto& p : op->parents) {
			if (p->requiresGrad) {
				ss << fmt::format("gradient = ");
				std::string comment;
				op->generateBwd(ss, i, old, *this, comment);
				ss << ";" << (comment.empty() ? "" : " //"+comment) << "\n";
				p->generateBackward(ss, visited);
			}
			++i;
		}
	}
}

void expr::countElems(nodeCountInfo& counter) const {
	++counter.nNodes;
	if(flags & constant)
		++counter.nConstants;
	if (flags & requiresGrad)
		++counter.nReqGrad;
	if (op)
		for (const auto& p : op->parents)
			p->countElems(counter);
}
int expr::getPrio() const {
	if (op)
		return abs(op->getPrio());
	return 999;
}
std::string expr::printExpr() const {
	if (op) {
		auto pl = op->parents[0];
		auto l = pl->printExpr(); if (pl->getPrio() <= getPrio()) l = bracket(l);
		std::string r;
		if (op->parents.size()>1) {
			auto pr = op->parents[1];
			r = pr->printExpr(); if (pr->getPrio() <= getPrio()) r = bracket(r);
		}
		return op->print(l, r);
	}
	else {
		//return std::format("{}",(void*)this);
		std::string s = getVarName();
		return s.empty() ? tostr(value) : s;
	}
}
