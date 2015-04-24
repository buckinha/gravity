from FireGirlOptimizer import *
from FireGirlStats import *

class FireGirlTests:
    #This class is designed to house functions useful in testing and analyzing the behavior
    #  and data of FireGirl objects, including when they are interacting with FireWoman data

    def __init__(self):

        #FLAG: general directive for tests to give a one-three line summary of their results
        self.PRINT_SUMMARIES = True
        #FLAG: in addition to printing summary information, ALSO print all details
        self.PRINT_DETAILS = True
        #FLAG: REGARDLESS OF OTHER FLAGS, print no ouptuts
        self.SILENT = False

    def stats_test(self, pathway_count = 20, years=100, start_ID=0):
        """Short test of the FireGirlStats functions

        Note: At the moment, a coin-toss policy is being used...
        """

        if not self.SILENT:
            if (self.PRINT_DETAILS or self.PRINT_SUMMARIES):
                print(" ")
                print("STATS TEST:")

        opt = FireGirlPolicyOptimizer()
        opt.SILENT = True
        opt.createFireGirlPathways(pathway_count, years, start_ID)

        #running stat functions
        sup_costs = suppression_cost_stats_by_year(opt.pathway_set)
        harvests = timber_harvest_stats_by_year(opt.pathway_set)
        growth = growth_stats_by_year(opt.pathway_set)
        fires = fire_stats_by_year(opt.pathway_set)


        if not self.SILENT:
            if self.PRINT_DETAILS:
                print("Suppression Cost Stats")
                print("ave, max, min, std, 95%high, 95%low")
                for i in range(len(sup_costs[0])):
                    #suppresion_cost_stats_by_year returns 6 lists
                    for j in range(6):
                        print(str(sup_costs[j][i]) + ","), #comma tells python not to end the line
                    #now end the line
                    print(" ")

                print(" ")
                print("Harvest Value Stats")
                print("ave, max, min, std, 95%high, 95%low")
                for i in range(len(harvests[0])):
                    #timber_harvest_stats_by_year returns 6 lists
                    for j in range(6):
                        print(str(harvests[j][i]) + ","), #comma tells python not to end the line
                    #now end the line
                    print(" ")

                print(" ")
                print("Timber Growth Stats")
                print("ave, max, min, std, 95%high, 95%low")
                for i in range(len(growth[0])):
                    #growth_stats_by_year returns 6 lists
                    for j in range(6):
                        print(str(growth[j][i]) + ","), #comma tells python not to end the line
                    #now end the line
                    print(" ")

                print(" ")
                print("Fire Stats:")
                print("Cells Burned, , , , , , Timber Lost, , , , , , Suppress Decisions")
                print("ave, max, min, std, 95%high, 95%low, ave, max, min, std, 95%high, 95%low")
                for i in range(len(fires[0][0])):
                    #fire_stats_by_year returns 3 lists
                    # the first is a sub-list of 6 stats for CELLS BURNED, as the features above

                    #printing Cells burned stats
                    for j in range(6):
                        print(str(fires[0][j][i]) + ","), #comma tells python not to end the line
                    #printing timber lost stats
                    for j in range(6):
                        print(str(fires[1][j][i]) + ","), #comma tells python not to end the line
                    #printing suppress decisions stat and ending the line
                    print(str(fires[2][i]))




    def monte_carlo_baselines(self, pathway_count=5, years=100, start_ID=2000):
        #This test will roll out N pathways using a let-burn, suppress-all, and coin-toss policies

        if not self.SILENT:
            if (self.PRINT_DETAILS or self.PRINT_SUMMARIES):
                print(" ")
                print("MONTE CARLO BASELINES:")

        opt = FireGirlPolicyOptimizer()
        pol = FireGirlPolicy()
        opt.SILENT = True


        #setting a let-burn policy
        pol.setLetBurn()

        net_val_sum = 0.0

        #doing let-burn pathways
        if not self.SILENT:
            if self.PRINT_DETAILS:
                print("--beginning let-burn pathway generation")

        for pw in range(pathway_count):
            opt.createFireGirlPathways(1, years, start_ID+pw, pol)
            #accessing the pathway's net value member directly
            net_val_sum += opt.pathway_set[0].net_value

        #calculate the average net value for all let-burn pathways
        ave_let_burn = net_val_sum / pathway_count

        #reset the sum for the next test
        net_val_sum = 0.0




        #setting a suppress-all policy
        pol.setSuppressAll()

        if not self.SILENT:
            if self.PRINT_DETAILS:
                print("--beginning suppress-all pathway generation")

        #doing suppress-all pathways
        for pw in range(pathway_count):
            opt.createFireGirlPathways(1, years, start_ID+pw, pol)
            #accessing the pathway's net value member directly
            net_val_sum += opt.pathway_set[0].net_value

        #calculate the average net value for all let-burn pathways
        ave_suppress_all = net_val_sum / pathway_count

        #reset the sum for the next test
        net_val_sum = 0.0




        #setting a coin-toss policy (all params = 0)
        pol.setParams([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

        if not self.SILENT:
            if self.PRINT_DETAILS:
                print("--beginning coin-toss pathway generation")

        #doing coin-toss pathways
        for pw in range(pathway_count):
            opt.createFireGirlPathways(1, years, start_ID+pw, pol)
            #accessing the pathway's net value member directly
            net_val_sum += opt.pathway_set[0].net_value

        #calculate the average net value for all let-burn pathways
        ave_coin_toss = net_val_sum / pathway_count


        if not self.SILENT:
            if (self.PRINT_DETAILS or self.PRINT_SUMMARIES):
                #printing summary information
                print(" ")
                print("Average Net Value of:")
                print("  let-burn pathways:     " + str(round(ave_let_burn)))
                print("  suppress-all pathways: " + str(round(ave_suppress_all)))
                print("  coin-toss pathways:    " + str(round(ave_coin_toss)))

            if self.PRINT_DETAILS:
                #printing details
                print("Test Details:")
                print("  Pathway Count: " + str(pathway_count))
                print("  Years/Pathway: " + str(years))
                print("  Pathway Start ID: " + str(start_ID))


        return [ave_let_burn, ave_suppress_all, ave_coin_toss]

class FireGirlTrials:
    """This class contains member functions that produce test data as outputs, replacing test scripts

    """

    def __init__():
        """Initialization function: Instantiates a FireGirlPolicyOptimizer and FireGirlPolicy for trials to use.

        """
        self.Opt = FireGirlPolicyOptimizer()
        self.Policy = FireGirlPolicy()







#setting up service-style tests. This will activate if you issue: "python FireGirlTests.py" at a command line, but
#  will otherwise be ignored
if __name__ == "__main__":


    tests = FireGirlTests()
    #tests.SILENT = True
    tests.PRINT_DETAILS = False
    tests.PRINT_SUMMARIES = True

    #running monte carlo tests, and ignoring results (at the moment)
    tests.monte_carlo_baselines()


    #finished
    print(" ")
    print(" ")
    print("All Tests Complete")
