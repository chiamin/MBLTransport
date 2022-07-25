#ifndef __INITSTATE_H_CMC__
#define __INITSTATE_H_CMC__
#include "SortBasis.h"

template <typename BasisL, typename BasisR, typename BasisS, typename SiteType>
MPS get_non_inter_ground_state (const BasisL& leadL, const BasisR& leadR, const BasisS& scatterer,
                                const SiteType& sites, Real muL, Real muS, Real muR, const ToGlobDict& to_glob)
{
    int N = to_glob.size();
    mycheck (length(sites) == N, "size not match");

    int Ns=0, Np=0;
    Real E = 0.;
    vector<string> state (N+1, "Emp");

    // Leads and scatterer
    auto occ_negative_en_states = [&to_glob, &state, &E, &Np, &Ns] (const auto& basis, Real mu)
    {
        string p = basis.name();
        for(int k = 1; k <= basis.size(); k++)
        {
            int i = to_glob.at({p,k});
            auto en = basis.en(k);
            if (en < mu)
            {
                state.at(i) = "Occ";
                E += en;
                Np++;
                if (p == "S")
                    Ns++;
            }
            else
            {
                state.at(i) = "Emp";
            }
        }
    };
    occ_negative_en_states (leadL, muL);
    occ_negative_en_states (leadR, muR);
    occ_negative_en_states (scatterer, muS);

    // Capacity site
    if (to_glob.count({"C",1}) != 0)
    {
        int j = to_glob.at({"C",1});
        state.at(j) = "0";
    }

    InitState init (sites);
    for(int i = 1; i <= N; i++)
        init.set (i, state.at(i));

    // Print information
    cout << "initial energy = " << E << endl;
    cout << "initial particle number = " << Np << endl;
    return MPS (init);
}
#endif
