#ifndef __HAMILTONIAN_H_CMC__
#define __HAMILTONIAN_H_CMC__

// C(i1,dag1) * C(i2,dag2) = \sum_k1 coef_i1,k1 C(k1,dag'1) * \sum_k2 coef_i2,k2 C(k2,dag'2)
// Return: vector of (coef, k1, dag'1, k2, dag'2)
template <typename Basis1, typename Basis2>
vector <tuple <Real,int,bool,int,bool>>
quadratic_operator_new (const Basis1& basis1, const Basis2& basis2, int i1, int i2, bool dag1, bool dag2, Real cutoff=1e-16)
{
    auto C1 = basis1.C_op (i1, dag1);     // i -> k, coef, dag
    auto C2 = basis2.C_op (i2, dag2);

    vector<tuple <Real,int,bool,int,bool>> ops;             // coef, k1, dag1, k2, dag2
    for(auto&& [k1,c1,dag1p] : C1)
    {
        for(auto&& [k2,c2,dag2p] : C2)
        {
            Real coef = c1*c2;
            if (abs(coef) > cutoff)
                ops.emplace_back (coef,k1,dag1p,k2,dag2p);    // Cdag_ki1 C_ki2
        }
    }
    return ops;
}

template <typename Basis1, typename Basis2, typename NumType>
void add_CdagC (AutoMPO& ampo, const Basis1& basis1, const Basis2& basis2, int i1, int i2, NumType coef, const ToGlobDict& to_glob)
{
    if (i1 < 0) i1 += basis1.size() + 1;
    if (i2 < 0) i2 += basis2.size() + 1;
    vector <tuple <Real,int,bool,int,bool>> terms = quadratic_operator_new (basis1, basis2, i1, i2, true, false);

    // 
    string p1 = basis1.name(),
           p2 = basis2.name();
    // Hopping terms
    for(auto [c12, k1, dag1, k2, dag2] : terms)  // coef, k1, dag1, k2, dag2
    {
        int j1 = to_glob.at({p1,k1});
        int j2 = to_glob.at({p2,k2});
        string op1 = (dag1 ? "Cdag" : "C");
        string op2 = (dag2 ? "Cdag" : "C");
        Real c = coef * c12;
        ampo += c, op1, j1, op2, j2;
    }
}

// Add -Delta C_i C_i+1 + h.c.
template <typename Basis1, typename Basis2, typename NumType>
void add_SC (AutoMPO& ampo, const Basis1& basis1, const Basis2& basis2, int i1, int i2, NumType Delta, const ToGlobDict& to_glob)
{
    if (i1 < 0) i1 += basis1.size()+1;
    if (i2 < 0) i2 += basis2.size()+1;
    vector <tuple <Real,int,bool,int,bool>> terms = quadratic_operator_new (basis1, basis2, i1, i2, false, false);

    string p1 = basis1.name(),
           p2 = basis2.name();
    for(auto [c12, k1, dag1, k2, dag2] : terms)  // coef, k1, dag1, k2, dag2
    {
        int j1 = to_glob.at({p1,k1});
        int j2 = to_glob.at({p2,k2});
        if (j1 != j2)
        {
            auto c = Delta * c12;
            auto cc = iut::conj (c);
            ampo += -c, "C", j1, "C", j2;
            ampo += -cc, "Cdag", j2, "Cdag", j1;
        }
    }
}

template <typename BasisL, typename BasisR, typename BasisS, typename SiteType>
AutoMPO get_ampo_tight_binding_NN_interaction
(const BasisL& leadL, const BasisR& leadR, const BasisS& scatterer, const SiteType& sites,
 Real mu_biasL, Real mu_biasR, Real tcL, Real tcR, Real V,
 const ToGlobDict& to_glob)
{
    mycheck (length(sites) == to_glob.size(), "size not match");

    AutoMPO ampo (sites);
    // Diagonal terms
    auto add_diag = [&ampo, &to_glob] (const auto& basis, Real mu)
    {
        string p = basis.name();
        for(int i = 1; i <= basis.size(); i++)
        {
            int j = to_glob.at({p,i});
            auto en = basis.en(i) - mu;
            ampo += en, "N", j;
        }
    };
    add_diag (leadL, mu_biasL);
    add_diag (leadR, mu_biasR);
    add_diag (scatterer, 0.);

    // Contact hopping
    add_CdagC (ampo, leadL, scatterer, -1, 1, -tcL, to_glob);
    add_CdagC (ampo, scatterer, leadL, 1, -1, -tcL, to_glob);
    add_CdagC (ampo, leadR, scatterer, 1, -1, -tcR, to_glob);
    add_CdagC (ampo, scatterer, leadR, -1, 1, -tcR, to_glob);

    // Nearest neighboring interaction
    if (V != 0.)
    {
        string sname = scatterer.name();
        for(int i = 1; i < scatterer.size(); i++)
        {
            int j1 = to_glob.at({sname,i});
            int j2 = to_glob.at({sname,i+1});
            ampo += V, "N", j1, "N", j2;
        }
    }
    return ampo;
}
#endif
