"""
PARAMETERS (see paper for details and references)

GT:
- Lifetime for AF = 40 years
- CAPEX: 1,175 USD/kW * 1.35 CAD/USD * 35,000 kW = CAD 55,518,750
- FIX. OPEX: 16.30 USD/kW-year * 1.35 CAD/USD * 35,000 kW = CAD 770,175  (Fix O&M. Note: Included in dynamic O&Ms?)
- VAR. OPEX: O&M Returned from D + Fuel costs returned from D
--> TOTEX GT = CAPEX * AF_GT + FIX. OPEX + VAR. OPEX

WT:
- Lifetime for AF = 25 years
- CAPEX: 1,265 USD/kW * 1.35 * 2,300 * numWT = CAD 3,927,825 * numWT
- OPEX: 26.34 USD/kW-year * 1.35 * 2,300 * numWT = CAD 81,786 * numWT  (fixed OPEX only)
--> TOTEX GT = CAPEX * AF_WIND + OPEX

BES:
- Lifetime for AF = 20 years
- CAPEX: 300,000 CAD/MWh * capBES (where 10% stem from inverters etc. for 4h BES)
- CAPEX (Formula): 270,000 CAD/MWh * capBES + 120,000 CAD/ MW rateBES
- OPEX: MAX(degradation cost from D, 3% capacity augmentation)
- OPEX (3% capacity augmentation): capBES * 270,000 CAD/MWh * 0.03
--> TOTEX BES = CAPEX * AF_BES + OPEX
"""


class ScoreConfig:
    """
    Class that scores different configurations of the GT-BES-WT island microgrid.

    To be initialized outside the sizing optimization loop with fixed parameters.
    Then, get_totex() is called inside the sizing optimization with the varying
    config parameters and returns the TOTEX of that configuration.
    """
    def __init__(
            self,
            discount_rate: float = 0.05,
            num_years: int = 1,  # Number of years modeled in the environment
            gt_capex_mw: float = 1_586_250,  # CAPEX per MW (CAD/MW)
            gt_opex_fix_mw_year: float = 22_005,  # Fixed OPEX per MW-year (CAD/MW-year)
            gt_lifetime: int = 40,  # Lifetime of the GT (years)
            gt_capacity_mw: float = 35,  # Installed capacity of the GT (MW)
            wt_capex_mw: float = 1_707_750,  # CAPEX per MW (CAD/MW)
            wt_opex_fix_mw_year: float = 35_559,  # Fixed OPEX per MW-year (CAD/MW-year)
            wt_lifetime: int = 25,  # Lifetime of the WT (years)
            wt_capacity_mw_turbine: float = 2.3,  # Capacity per turbine (MW)
            bes_capex_per_mwh: float = 270_000,  # CAPEX per MWh (CAD/MWh)
            bes_capex_rate_per_mw: float = 120_000,  # CAPEX per MW (CAD/MW)
            bes_lifetime: int = 20,  # Lifetime of the BES (years)
    ):
        """
        Initialize with fixed inputs for GT, WT, and BES components.

        Parameters:
        - discount_rate: Shared discount rate for financial calculations.
        - self.num_years = num_years  # Add the number of years modeled


        GT Parameters:
        - gt_capex_mw: CAPEX per MW (CAD/MW).
        - gt_opex_fix_mw_year: Fixed OPEX per MW-year (CAD/MW-year).
        - gt_lifetime: Lifetime of the GT (years).
        - gt_capacity_mw: Installed capacity of the GT (MW).

        WT Parameters:
        - wt_capex_mw: CAPEX per MW (CAD/MW).
        - wt_opex_fix_mw_year: Fixed OPEX per MW-year (CAD/MW-year).
        - wt_lifetime: Lifetime of the WT (years).
        - wt_capacity_mw_turbine: Capacity per turbine (MW).

        BES Parameters:
        - bes_capex_per_mwh: CAPEX per MWh (CAD/MWh).
        - bes_capex_rate_per_mw: CAPEX per MW (CAD/MW).
        - bes_lifetime: Lifetime of the BES (years).
        """
        self.discount_rate = discount_rate
        self.num_years = num_years  # Add the number of years modeled

        # GT Parameters
        self.gt_capex_mw = gt_capex_mw
        self.gt_opex_fix_mw_year = gt_opex_fix_mw_year
        self.gt_lifetime = gt_lifetime
        self.gt_capacity_mw = gt_capacity_mw
        self.af_gt = self._compute_annuity_factor(gt_lifetime)

        # WT Parameters
        self.wt_capex_mw = wt_capex_mw
        self.wt_opex_fix_mw_year = wt_opex_fix_mw_year
        self.wt_lifetime = wt_lifetime
        self.wt_capacity_mw_turbine = wt_capacity_mw_turbine
        self.af_wt = self._compute_annuity_factor(wt_lifetime)

        # BES Parameters
        self.bes_capex_per_mwh = bes_capex_per_mwh
        self.bes_capex_rate_per_mw = bes_capex_rate_per_mw
        self.bes_lifetime = bes_lifetime
        self.af_bes = self._compute_annuity_factor(bes_lifetime)

    def _compute_annuity_factor(self, lifetime: int) -> float:
        """
        Compute annuity factor based on discount rate and component lifetime.
        Formula: (r * (1 + r)^n) / ((1 + r)^n - 1)
        """
        r = self.discount_rate
        n = lifetime
        return (r * (1 + r) ** n) / ((1 + r) ** n - 1)

    def get_totex(
            self,
            numWT: int,  # Number of wind turbines
            capBES: float,  # BES energy capacity (MWh)
            rateBES: float,  # BES charge/discharge rate (MW)
            fuel_cost: float,  # Fuel cost for GT (CAD)
            variable_om: float,  # Variable O&M cost for GT (CAD)
            degradation_cost: float,  # BES degradation cost (CAD)
    ) -> float:
        """
        Compute TOTEX for a given configuration and dispatch policy.

        Parameters:
        - numWT: Number of wind turbines.
        - capBES: BES energy capacity (MWh).
        - rateBES: BES charge/discharge rate (MW).
        - fuel_cost: Fuel cost for GT (CAD).
        - variable_om: Variable O&M cost for GT (CAD).
        - degradation_cost: BES degradation cost (CAD).

        Returns:
        - totex: Total expenditure (CAD).
        """
        # Compute TOTEX for each component
        totex_gt = self._compute_totex_gt(fuel_cost, variable_om)
        totex_wt = self._compute_totex_wt(numWT)
        totex_bes = self._compute_totex_bes(capBES, rateBES, degradation_cost)

        # Total TOTEX
        return totex_gt + totex_wt + totex_bes

    def _compute_totex_gt(self, fuel_cost: float, variable_om: float) -> float:
        """
        Compute TOTEX for the GT component.

        Parameters:
        - fuel_cost: Fuel cost for GT (CAD).
        - variable_om: Variable O&M cost for GT (CAD).

        Returns:
        - totex_gt: Total expenditure for GT (CAD).
        """
        capex = self.gt_capex_mw * self.gt_capacity_mw
        opex_fix = self.gt_opex_fix_mw_year * self.gt_capacity_mw
        var_opex = (fuel_cost + variable_om) / self.num_years
        return capex * self.af_gt + opex_fix + var_opex

    def _compute_totex_wt(self, numWT: int) -> float:
        """
        Compute TOTEX for the WT component.

        Parameters:
        - numWT: Number of wind turbines.

        Returns:
        - totex_wt: Total expenditure for WT (CAD).
        """
        capex = self.wt_capex_mw * self.wt_capacity_mw_turbine * numWT
        opex = self.wt_opex_fix_mw_year * self.wt_capacity_mw_turbine * numWT
        return capex * self.af_wt + opex

    def _compute_totex_bes(self, capBES: float, rateBES: float, degradation_cost: float) -> float:
        """
        Compute TOTEX for the BES component.

        Parameters:
        - capBES: BES energy capacity (MWh).
        - rateBES: BES charge/discharge rate (MW).
        - degradation_cost: BES degradation cost (CAD).

        Returns:
        - totex_bes: Total expenditure for BES (CAD).
        """
        capex = (self.bes_capex_per_mwh * capBES) + (self.bes_capex_rate_per_mw * rateBES)

        if capBES == 0:  # Handle no BES situation
            return 0.0

        opex = max(degradation_cost / self.num_years, capBES * self.bes_capex_per_mwh * 0.03)
        return capex * self.af_bes + opex
