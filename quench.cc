#include <iomanip>
#include "itensor/all.h"
#include "Timer.h"
Timers timer;
#include "ReadInput.h"
#include "IUtility.h"
#include "MyObserver.h"
#include "TDVPObserver.h"
#include "tdvp.h"
#include "basisextension.h"
#include "InitState.h"
#include "Hamiltonian.h"
#include "ReadWriteFile.h"
#include "OneParticleBasis.h"
using namespace itensor;
using namespace std;

void print_orbs (const vector<SortInfo>& orbs)
{
    cout << "Orbitals: name, ki, energy" << endl;
    for(int i = 1; i <= orbs.size(); i++)
    {
        auto [name, ki, en] = orbs.at(i-1);
        cout << i << ": " << name << " " << ki << " " << en << endl;
    }
}

template <typename Basis1, typename Basis2, typename SiteType>
MPO get_current_mpo (const SiteType& sites, const Basis1& basis1, const Basis2& basis2, int i1, int i2, const ToGlobDict& to_glob)
{
    AutoMPO ampo (sites);
    add_CdagC (ampo, basis1, basis2, i1, i2, 1., to_glob);
    auto mpo = toMPO (ampo);
    return mpo;
}

inline Real get_current (const MPO& JMPO, const MPS& psi)
{
    auto J = innerC (psi, JMPO, psi);
    return -2. * imag(J);
}

tuple<int,int> find_scatterer_region (const ToLocDict& to_loc, string Sname, string Cname, bool include_charge=true)
{
    bool start_scatter = false;
    int i1 = -1,
        i2 = to_loc.size();
    for(int i = 1; i < to_loc.size(); i++)
    {
        auto [name, ind] = to_loc.at(i);
        bool is_scatter = (name == Sname || (include_charge and name == Cname));
        if (!start_scatter and is_scatter)
        {
            start_scatter = true;
            i1 = i;
        }
        if (start_scatter and !is_scatter)
        {
            i2 = i-1;
            break;
        }
    }
    mycheck (i1 > 0, "Do not search a system site");

    // Check all the scatterer sites are together
    for(int i = i2+1; i < to_loc.size(); i++)
    {
        auto [name, ind] = to_loc.at(i);
        bool is_scatter = (name == Sname || (include_charge and name == Cname));
        mycheck (!is_scatter, "Scatterer sites are not all together");
    }
    return {i1, i2};
}

int main(int argc, char* argv[])
{
    string infile = argv[1];
    InputGroup input (infile,"basic");

    auto dt            = input.getReal("dt");
    auto time_steps    = input.getInt("time_steps");
    auto quench_type   = input.getString("quench_type");
    auto seed          = input.getInt("seed",time(NULL));

    auto NumCenter           = input.getInt("NumCenter");
    auto mixNumCenter        = input.getYesNo("mixNumCenter",false);
    auto globExpanNStr       = input.getString("globExpanN","inf");
    int globExpanN;
    if (globExpanNStr == "inf" or globExpanNStr == "Inf" or globExpanNStr == "INF")
        globExpanN = std::numeric_limits<int>::max();
    else
        globExpanN = std::stoi (globExpanNStr);
    auto globExpanItv        = input.getInt("globExpanItv",1);
    auto globExpanCutoff     = input.getReal("globExpanCutoff",1e-8);
    auto globExpanKrylovDim  = input.getInt("globExpanKrylovDim",3);
    auto globExpanHpsiCutoff = input.getReal("globExpanHpsiCutoff",1e-8);
    auto globExpanHpsiMaxDim = input.getInt("globExpanHpsiMaxDim",300);
    auto globExpanMethod     = input.getString("globExpanMethod","DensityMatrix");
    auto UseSVD              = input.getYesNo("UseSVD",true);
    auto SVDmethod           = input.getString("SVDMethod","gesdd");  // can be also "ITensor"
    auto WriteDim            = input.getInt("WriteDim");

    auto measure_entropy = input.getYesNo("measure_entropy");
    auto measure_entropy_cutoff = input.getReal("measure_entropy_cutoff",1e-14);
    auto measure_entropy_maxdim = input.getInt("measure_entropy_maxdim",std::numeric_limits<int>::max());

    auto write         = input.getYesNo("write",false);
    auto write_dir     = input.getString("write_dir",".");
    auto write_file    = input.getString("write_file","");
    auto read          = input.getYesNo("read",false);
    auto read_dir      = input.getString("read_dir",".");
    auto read_file     = input.getString("read_file","");

    auto sweeps        = iut::Read_sweeps (infile, "TDVP");

    cout << setprecision(14) << endl;

    MPS psi;
    MPO H;
    // Define 
    int step = 1;
    auto sites = Fermion();
    Args args_basis;

    ToGlobDict to_glob;
    ToLocDict to_loc;
    OneParticleBasis leadL, leadR, charge;
    OneParticleBasis scatterer;

    // -- Initialization --
    if (!read)
    {
        auto L_lead     = input.getInt("L_lead");
        auto L_device   = input.getInt("L_device");
        auto t_lead     = input.getReal("t_lead");
        auto t_device   = input.getReal("t_device");
        auto t_contactL = input.getReal("t_contactL");
        auto t_contactR = input.getReal("t_contactR");
        auto mu_leadL   = input.getReal("mu_leadL");
        auto mu_leadR   = input.getReal("mu_leadR");
        auto W_device   = input.getReal("mu_device_disorder_strength");
        auto mu_biasL   = input.getReal("mu_biasL");
        auto mu_biasS   = input.getReal("mu_biasS");
        auto mu_biasR   = input.getReal("mu_biasR");
        auto V_device   = input.getReal("V_device");
        auto damp_decay_length = input.getInt("damp_decay_length",0);
        auto sweeps_DMRG = iut::Read_sweeps (infile, "DMRG");

        // Factor for exponentially decaying hoppings
        Real damp_fac = (damp_decay_length == 0 ? 1. : exp(-1./damp_decay_length));
        // Create bases for the leads
        cout << "H left lead" << endl;
        leadL = OneParticleBasis ("L", L_lead, t_lead, mu_leadL, damp_fac, true, true);
        cout << "H right lead" << endl;
        leadR = OneParticleBasis ("R", L_lead, t_lead, mu_leadR, damp_fac, false, true);
        // Create basis for scatterer
        cout << "H dev" << endl;
        vector<Real> mus;
        {
        // Generate random potential
            //std::random_device rd;
            std::mt19937 rgen(seed);
            std::uniform_real_distribution<> dist (-W_device,W_device); // distribution in range [-W_device, W_device]
            cout << "Disordered chemical potential" << endl;
            for(int i = 0; i < L_device; i++)
            {
                auto rand = dist (rgen);
                mus.push_back (rand);
                cout << i+1 << " " << rand << endl;
            }
        }
        Matrix Hdev = tight_binding_Hamilt (L_device, t_device, mus);
        scatterer = OneParticleBasis ("S", Hdev);

        // Combine and sort all the basis states
        auto info = sort_by_energy_S_middle (scatterer, leadL, leadR);
        tie(to_glob, to_loc) = make_orb_dicts (info);
        print_orbs(info);

        // Make SiteSet
        int N = to_glob.size();
        args_basis = {"ConserveQNs",true};
        sites = Fermion (N, args_basis);

        // Make initial Hamiltonian MPO
        // 1. mu quench
        if (quench_type == "mu_quench")
        {
            // Hamiltonian MPO H0 for initial state
            // H0: No bias potential
            auto ampoi = get_ampo_tight_binding_NN_interaction (leadL, leadR, scatterer, sites, 0., 0., t_contactL, t_contactR, V_device, to_glob);
            auto Hi = toMPO (ampoi);
            cout << "Initial MPO dim = " << maxLinkDim(Hi) << endl;

            // Initialze MPS which is the ground state of H0
            psi = get_non_inter_ground_state (leadL, leadR, scatterer, sites, 0., 0., 0., to_glob);   // disconnected product state
            psi.position(1);
            itensor::Real en0 = dmrg (psi, Hi, sweeps_DMRG);
            cout << "Initial state bond dim = " << maxLinkDim(psi) << endl;
            cout << "Initial energy = " << inner (psi,Hi,psi) << endl;

            // Make Hamiltonian MPO H for time evolution
            // H: Applying bias potential
            auto ampo = get_ampo_tight_binding_NN_interaction (leadL, leadR, scatterer, sites, mu_biasL, mu_biasR, t_contactL, t_contactR, V_device, to_glob);
            H = toMPO (ampo);
            cout << "MPO dim = " << maxLinkDim(H) << endl;
        }
        // 2. Density quench
        else if (quench_type == "density_quench")
        {
            // Initialze MPS psi
            // psi: Ground state of disconnected leads and scatterer with bias potentials
            psi = get_non_inter_ground_state (leadL, leadR, scatterer, sites, mu_biasL, mu_biasS, mu_biasR, to_glob);
            psi.position(1);

            // Make Hamiltonian MPO H
            // H: Connected system with no bias potential
            auto ampo = get_ampo_tight_binding_NN_interaction (leadL, leadR, scatterer, sites, 0., 0., t_contactL, t_contactR, V_device, to_glob);
            H = toMPO (ampo);
            cout << "MPO dim = " << maxLinkDim(H) << endl;
            cout << "Initial energy = " << inner (psi,H,psi) << endl;
        }
        else
        {
            cout << "Unknown quench type: " << quench_type << endl;
            throw;
        }
    }
    else
    {
        ifstream ifs = open_file (read_dir+"/"+read_file);
        iut::read_all (ifs, psi, H, args_basis, step, to_glob, to_loc, leadL, leadR, scatterer, charge);
        sites = Fermion (siteInds(psi));
    }
    // -- End of initialization --


    // -- Observer --
    auto obs = TDVPObserver (sites, psi);
    // Current MPO
    auto jmpoL = get_current_mpo (sites, leadL, leadL, -2, -1, to_glob);
    auto jmpoR = get_current_mpo (sites, leadR, leadR, 1, 2, to_glob);

    // Find the scatterer region, which will be used in compaute the entanglement entropy
    // (Assume that all the scatterer sites are together; otherwise raise error.)
    auto [si1, si2] = find_scatterer_region (to_loc, scatterer.name(), charge.name());
    cout << "charge and scatterer is between sites " << si1 << " " << si2 << endl;

    // -- Time evolution --
    cout << "Start time evolution" << endl;
    cout << sweeps << endl;
    psi.position(1);
    Real en, err;
    Args args_tdvp_expansion = {"Cutoff",globExpanCutoff, "Method","DensityMatrix",
                                "KrylovOrd",globExpanKrylovDim, "DoNormalize",true, "Quiet",true};
    Args args_tdvp  = {"Quiet",true,"NumCenter",NumCenter,"DoNormalize",true,"Truncate",true,
                       "UseSVD",UseSVD,"SVDmethod",SVDmethod,"WriteDim",WriteDim,"mixNumCenter",mixNumCenter};
    LocalMPO PH (H, args_tdvp);
    while (step <= time_steps)
    {
        cout << "step = " << step << endl;

        // Subspace expansion
        if (maxLinkDim(psi) < sweeps.mindim(1) or (step < globExpanN and (step-1) % globExpanItv == 0))
        {
            timer["glob expan"].start();
            addBasis (psi, H, globExpanHpsiCutoff, globExpanHpsiMaxDim, args_tdvp_expansion);
            PH.reset();
            timer["glob expan"].stop();
        }

        // Time evolution
        timer["tdvp"].start();
        //tdvp (psi, H, 1_i*dt, sweeps, obs, args_tdvp);
        TDVPWorker (psi, PH, 1_i*dt, sweeps, obs, args_tdvp);
        timer["tdvp"].stop();
        auto d1 = maxLinkDim(psi);

        // Measure currents by MPO
        timer["current mps"].start();
        auto jL = get_current (jmpoL, psi);
        auto jR = get_current (jmpoR, psi);
        cout << "\tI L/R = " << jL << " " << jR << endl;
        timer["current mps"].stop();

        // Measure entanglement entropy
        if (measure_entropy)
        {
            timer["entang entropy"].start();
            auto EEs = get_entang_entropy (psi, si1, si2, {"Cutoff",measure_entropy_cutoff,"MaxDim",measure_entropy_maxdim});
            timer["entang entropy"].stop();
            cout << "\tEE = " << EEs << endl;
//            for(int i = 0; i < EEs.size(); i++)
//                cout << "\tEE " << i << " = " << EEs.at(i) << endl;
        }

        step++;
        if (write)
        {
            timer["write"].start();
            ofstream ofs (write_dir+"/"+write_file);
            iut::write_all (ofs, psi, H, args_basis, step, to_glob, to_loc, leadL, leadR, scatterer, charge);
            timer["write"].stop();
        }
    }
    timer.print();
    return 0;
}
