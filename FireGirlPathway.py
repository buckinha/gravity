import random, math, tempfile
from FireGirl_DS_alg import *
from FireGirlPolicy import *
from FireGirlPathwayLogbook import *
from PIL import Image #for landscape snapshot creation
import growth_models #importing arrays that define timber value, indexed by age

class FireGirlPathway:
    #This class holds a single FireGirl pathway, and has all the necessary 
    #  functions to allow it to evolve, change, etc..., as well as to read/write
    #  the data to file.
    
    ###FEATURE VALUES ###
    # For the suppression rule in FireGirl pathways, the following are the 
    #     features being used:
    #
    #  Wind Speed - one value per day/fire
    #  Temperature - one value per day/fire
    #  Date of Ignition
    #  (Date of Ignition)^2
    #  Timber Value at ignition point
    #  Average Timber Value adjacent 8
    #  Average Timber Value adjacent 24
    #  Fuel Load at ignition point
    #  Average Fuel Load adjacent 8
    #  Average Fuel Load adjacent 24
    
    def __init__(self, ID_number, policy_object=None, FIREGIRL_DATA=True):
        #The ID of the pathway is used to uniquely identify it for file in/out 
        # functions, and to seed it's random number generation procedures, for 
        # replicability

        #assigning the ID number (required!)
        self.ID_number = ID_number
        
        #for testing only, a flag for Debug outputs
        self.DEBUG = False
        
        #A flag for using 5-15-15 bugs for the FIREvis paper
        self.USE_BUGS = False

        #Flag: if True, suppress all print statements
        self.SILENT = False
        
        #The policy object is defined in FireGirl_Policy.py, and contains everything
        #  required to evaluate a set of features against a given policy. To the
        #  pathway object, this operates as a black box.

        self.Policy = None
        if policy_object == None:
            #No policy was given, so create a new one:
            if FIREGIRL_DATA == True:
                #This is a FireGirl pathway, so make a new policy with 11 parameters
                self.Policy = FireGirlPolicy(None,0,11)
            else:
                #This is a FireWoman pathway, so make a new policy appropriately...
                self.Policy = FireGirlPolicy(None,0,50) #TODO: HOW MANY FOR FIREWOMAN?
        else:
            #A policy was passed to the constructor, so just use it and assume the user
            # set the appropriate number of parameters, etc...
            self.Policy = policy_object


        # A list of each ignition event, recorded as FireGirlIgnitonRecord objects
        self.ignition_events = []

        # A value to hold the net value of the pathway after all fires, suppresions, 
        #    etc...
        self.net_value = 0

        #In order to use truly pseudo-random numbers throughout, this flag can be 
        #  set to True. This will eliminate the option of replicability, so it
        #  defaults to False
        self.TRUE_PSEUDO_RANDOM = False

        #Flag: Use log(probabilities)  -  If we want to force sums of log(probs), set to True
        #                                 To just multiply probabilities, set to False
        self.USE_LOG_PROB = False

        
        #The current pathway year starts at 0 and then increments as needed.
        #  It is used, along with the pathway's ID number, to seed the random
        #  number generator. Mainly this is because replicability needs to be 
        #  retained even if the pathway is saved, loaded, etc... without repeating
        #  numbers already drawn.
        self.year = 0
        
        #A flag that indicates whether or not this pathway contains FireGirl Data
        #   Set True if so, and False if its using FireWoman data
        self.FIREGIRL_DATA = FIREGIRL_DATA

        
        ###########################
        # FIRGIRL MODEL VARIABLES #
        ###########################

        if self.FIREGIRL_DATA == True:

            #Each cell has it's own stand values. So far, just timber_value, representing
            #  and integrated measure of age/density of harvestable trees, and fuel_load
            #  which is a more abstract interpretation of the dog-hair thickets, downed
            #  snags, etc... that build up when there isn't fire. It will be a primary
            #  determinant of whether a fire burns into the crowns, determining the 
            #  timber_value that is lost, or not, by the blaze.
                    
            #The Logbook object allows this pathway to record its yearly history
            #self.Logbook = FireGirlPathwayLogbook()

            #Starting a list to hold FireGirl_FireLog objects. Each one holds a full 
            #  record of one fire, including the cells that burn, when they burn, and 
            #  other information like crownfires.
            self.FireLog = []


            #The width and height are being set to 129 to correspond with the diamond-
            #  -square algorithm outputs. The centered "window of interest" will be
            #  the central 43x43 square
            self.width = 129
            self.height = 129

            #defining variable window-of-interest sizes
            self.window_NW = 43
            self.window_SE = 86 

            #lists to remember past fuel load and timber value grids
            self.SAVE_HISTORY = False
            self.timber_value_history = []
            self.fuel_load_history = []
            
            #Creating a rectangular array to hold timber values for each cell
            # self.timber_value = []
            # for i in range(self.width):
                # self.timber_value.append([])
                # for j in range(self.height):
                    # self.timber_value[i].append(0)
            #self.timber_value = [[0]*self.height]*self.width
            
            #Creating a rectangular array to hold fuel-load values for each cell
            # self.fuel_load = []
            # for i in range(self.width):
                # self.fuel_load.append([])
                # for j in range(self.height):
                    # self.fuel_load[i].append(0)
            #self.fuel_load = [[0]*self.height]*self.width

            self.timber_value = [None]*self.width
            self.fuel_load = [None]*self.width
            self.stand_age = [None]*self.width
            self.fuel_age = [None]*self.width
            for i in range(self.width):
                self.timber_value[i] = [0]*self.height
                self.fuel_load[i] = [0]*self.height
                self.stand_age[i] = [0]*self.height
                self.fuel_age[i] = [0]*self.height
            
            #Creating a rectangular array to hold the stand age of the cell
            # self.stand_age = []
            # for i in range(self.width):
                # self.stand_age.append([])
                # for j in range(self.height):
                    # self.stand_age[i].append(0)
            #self.stand_age = [[0]*self.height]*self.width
            
            #Creating an array to hold fuel age
            # self.fuel_age = []
            # for i in range(self.width):
                # self.fuel_age.append([])
                # for j in range(self.height):
                    # self.fuel_age[i].append(0)
            #self.fuel_age = [[0]*self.height]*self.width
            
            # ignition probability: the likelihood of there being an important fire on
            #   any given year
            self.ignition_prob = 100
            
            # temperature variables: These variables control the temperature average and
            #    mean, throughout the year. They are based on a cosine distribution (for
            #    smoothness).  Positive temperatures encourage firespread, while negative
            #    temperatures discourage firespread
            self.temp_summer_high = 90
            self.temp_winter_low = -90
            #for the moment, I'm assuming Jan 1 (day 0) is the coldest, on average)
            #  and whatever the middle date is (day 183) is the hotest, on average)
            #
            #also, for the moment, i'm leaving variance constant
            self.temp_var = 10
            
            # wind variables: These variables control the windspeed average and mean
            #   throughout the year.
            #For the moment, I'm just having it draw from an exponential distribution
            #Windspeeds have a mean of 10 units, but can get much higher, on occasion,
            # because of the exponential
            self.wind_mean = 10
            
            ### FIRE MODEL ###
            #These are the parameters that give shape to the fire spreadrate calculation
            # It is a logistic function that takes FireGirl's (windspeed + temperature)
            # as it's input.
            self.fire_param_inputscale = 10
            self.fire_param_outputscale = 10
            self.fire_param_zeroadjust = 15
            self.fire_param_smoothness = 0.4
            
            #when doing the priority queue fire model, we have to decide how many cells
            #  around the current cell to calculate fire arrivals to. This can be any 
            #  integer between 1 and 43.
            self.fire_param_reach = 1
            
            #Minimum Spread Restrictions
            # if (wind + temp) fall below a certain threshold, there will not be any
            #   fire spread.  If there's not enough fuel to sustain a fire, there will 
            #   not be any fire spread
            self.min_spread_windtemp = 0 # this enforces a rule that w+t is positive.
            self.min_spread_fuel = 10 #fuel ranges from 0 to 100, so this is the very
                                       #  lowest range.
            
            
            #These are the parameters governing the logistic function that determines
            #  what percentage of the timber_value is susceptible to burning. In this 
            #  model, that percentage corresponds to the total timber value that can be
            #  lost during a fire.
            #For now, just multiplying between one fifth and one tenth of the spread rate
            #  by this percentage could yield a decent loss value.
            self.crownfire_param_inputscale = 10
            self.crownfire_param_outputscale = 1
            self.crownfire_param_zeroadjust = 5
            self.crownfire_param_smoothness = 1



            #### SUPPRESSION MODEL
            # a list to hold each year's suppression costs
            self.yearly_suppression_costs = []

            #for the end-of-fire time, this is the average "days" it takes. This actually
            # effects the time that will be allowed to the fire spreading model before it cuts itself
            # off.
            self.fire_average_end_day = 2

            #How much does suppression reduce the fire spread rate:
            self.fire_suppression_rate = 0.05

            #how much does it cost to suppress a fire?
            self.fire_suppression_cost_per_cell = 400 #first runs were 10
            self.fire_suppression_cost_per_day = 2000 #            and 50



            #### TREE GROWTH MODEL ####
            self.growth_timber_constant = 22.0
            self.growth_fuel_accumulation = 2

            #a list to hold pre-calculated values for growth.
            self.growth_model_1 = growth_models.growth_model_1
            self.growth_model_2 = growth_models.growth_model_2
            #this is numeric switch, to define which growth model is used by this pathway
            self.using_growth_model = 2 
            self.growth_model = self.growth_model_2


            #a list to hold each year's total pathway growth amount
            self.yearly_growth_totals = []


            #### LOGGING MODEL ####
            #a list to hold each year's logging total
            self.yearly_logging_totals = []


            #logging site block width
            self.logging_block_width = 10
            #the logging model will not cut stands under this timber value
            self.logging_min_value = 50  #30-yr-old stands should have a value of about 75
            #the logging model will leave this much fuel_load as "slash" after it cuts a stand
            self.logging_slash_remaining = 10
            #the logging model will cut this percent of the years total timber growth
            self.logging_percentOfIncrement = 0.95
            #how many cuts can be made in the search for this years timber allowance?
            self.logging_max_cuts = 30



    #####################
    # GENERAL FUNCTIONS #
    #####################

    def chooseSuppression(self, suppress_prob):
        #This function just rolls the dice against the given probability and
        # returns true for suppress, and false for let-burn
        
        #TODO enforce TRUE_PSEUDO_RANDOM flag
        if random.uniform(0,1) < suppress_prob:
            return True
        else:
            return False
    
    def getIgnitionCount(self):
        return len(self.ignition_events)

    def getProb(self, ignition_index):
        #This function returns it's current Policy's probability calculation for
        #  the ignition at the given index
        f = self.ignition_events[ignition_index].getFeatures()
        return self.Policy.calcProb(f)

    def getChoice(self, ignition_index):
        #the choice for any given ignition never changes.
        return self.ignition_events[ignition_index].getChoice()

    def getCrossProduct(self, ignition_index):
        #This function returns it's current Policy's crossproduct calculation for
        #  the ignition at the given index
        f = self.ignition_events[ignition_index].getFeatures()
        return self.Policy.crossProduct(f)

    def getFeature(self, ignition_index, k):
        #this function returns the kth feature of the ith ignition

        #checking bounds
        if ignition_index >= len(self.ignition_events):
            print("Error in FGPathway.getFeature(i,k): There is no ignition at index i")
        else:
            if k >= len(self.ignition_events[ignition_index].getFeatures()):
                print("Error in FGPathway.getFeature(i,k): There is no feature at index k")
                print(" -feature list has " + str(len(self.ignition_events[ignition_index].getFeatures())) + " elements.")
                print(" -function call requesting element at index " + str(k))


        return self.ignition_events[ignition_index].getFeatures()[k]


    def assignPolicy(self, policy):
        self.Policy = policy

    def resetPolicy(self):
        self.Policy = FireGirlPolicy()
        
    def calcTotalProb(self):
        #This function looks through each ignition event and computes the 
        #   product of all the suppression/let-burn probabilities.  If the
        #   USE_LOG_PROB flag is set, it will sum the logged probabilities
        #   instead.

        total_prob = 0

        if self.USE_LOG_PROB == False:
            #according to the flag, we're not using sum(log(probs)) so just
            #  compute the product

            product = 1.0
            break_loop = False
            
            if self.DEBUG == True:
                print("In ls.calcTotalProb()...  Pathway " + str(self.ID_number))
                
            for ign in self.ignition_events:
                if self.DEBUG == True:
                    print("  ign " + str(self.ignition_events.index(ign)))

                try:
                    #use the current policy to calculate a new probability with the original features
                    #   of each ignition
                    p = self.Policy.calcProb(ign.getFeatures())
                      
                    p_actual = 0.0
                    if ign.getChoice() == True:
                        #this fire was suppressed, so use the probability as is
                        p_actual = p
                    else:
                        #this fire was allowed to burn, so use the other probability
                        p_actual = 1.0 - p
                        
                    if self.DEBUG == True:
                        print("calculated: " + str(p) + "  actual: " + str(p_actual))
                        
                    product *= p_actual
                    
                except (TypeError):
                    print("FGPathway.calcTotalProb() encountered a TypeError:")
                    print(" ignition.getProb() returns: " + str(ign.getProb()))
                    print(" ignition.features are:")
                    print(ign.features)
                except (ArithmeticError):
                    #it is possible that a long series of multiplications over fractions
                    #  could result in an underflow conidtion. Here I'm assuming that is
                    #  the type of ArithmeticError that we've run into.

                    #since the value underflowed, it is clearly VERY small, so just
                    # set it to zero
                    product = 0.0

                    #and report it
                    print("a FireGirlPathway object reports an underflow condition during a PRODUCT calculation")

                    #and don't bother with any more multiplications
                    break_loop = True

                if break_loop == True: break 

            total_prob = product

        else:
            #according to the flag, we ARE using sum(log(probs)) so do so:
            
            summation = 0.0

            for ign in self.ignition_events:
                #use the current policy to calculate a new probability with the original features
                #   of each ignition
                p = self.Policy.calcProb(ign.getFeatures())
                p_actual = 0.0
                if ign.getChoice() == True:
                    #this fire was suppressed, so use the probability as is
                    p_actual = p
                else:
                    #this fire was allowed to burn, so use the other probability
                    p_actual = 1.0 - p
                    
                summation += math.ln(p_actual)

            try:
                total_prob = math.exp(summation)
            except (ArithmeticError):
                #I'm assuming here that an underflow condition is the only type of
                #  arithmeticError I'm likely to find...

                #If it underflows, the probability total is very small, so just set
                #  it to zero
                total_prob = 0.0

                #and report it
                print("a FireGirlPathway object reports an underflow condition during a SUMMATION calculation")


        return total_prob
    
    def calcAveProb(self):
        #This function returns the average probability of each of it's ignition events, given the 
        # policy currently set.
        
        sum_of_probs = 0
        
        #for use within the loop
        p = 0
        
        for ign in self.ignition_events:
            #use the current policy to calculate a new probability with the original features
            #   of each ignition
            p = self.Policy.calcProb(ign.getFeatures())
              
            p_actual = 0.0
            if ign.getChoice() == True:
                #this fire was suppressed, so use the probability as is
                p_actual = p
            else:
                #this fire was allowed to burn, so use the other probability
                p_actual = 1.0 - p
            
            sum_of_probs += p_actual
        
        #now that we've summed all the probabilities, divide by the total number of ignitions
        ave_prob = sum_of_probs / len(self.ignition_events)
        
        return ave_prob
    
    def calcSumOfProbs(self):
        #this function simply returns the sums of the probabilities for each ignition in the chain,
        # for use by FireGirlOptimizer's probability normalization calculations
        
        sum_of_probs = 0
        
        #for use within the loop
        p = 0
        
        for ign in self.ignition_events:
            #use the current policy to calculate a new probability with the original features
            #   of each ignition
            p = self.Policy.calcProb(ign.getFeatures())
              
            p_actual = 0.0
            if ign.getChoice() == True:
                #this fire was suppressed, so use the probability as is
                p_actual = p
            else:
                #this fire was allowed to burn, so use the other probability
                p_actual = 1.0 - p
            
            sum_of_probs += p_actual
        
        
        return sum_of_probs
    
    def getNetValue(self):
        return self.net_value

    def setYear(self, year):
        #this is intended to be used when loading data from saved FireGirl or FireWoman
        #  data, etc...  FireGirl pathways will update it themselves when they're evolving.
        self.year = int(year)
    
    
    ###############################
    # FireGirl-specific Functions #
    ###############################

    def setLandscapeSize(self,size):
        """Sets the usable landscape size to the given dimensions.
        Note: The diamond-square algorithm still creates the full 129x129 grid, 
        but all simulations will only use the NW subset of that (still a square)

        Note: This function will also set the "window of interest" to be ~1/3 of the size 
        of the whole, centered (with rounding) in the middle.

        Arguements
        size: integer: accepts vals betwen 9 and 129. Anything larger or smaller will be corrected. 
        -  both landscape width AND height will be set to this value.
        """
        #constaining input values
        if size < 9: size = 9
        if size > 129: size = 129

        #setting window of interest
        wind_nw = int(round(0.3333 * size))
        wind_se = int(round(0.6666 * size))

        #setting values
        self.width = size
        self.height = size
        self.window_NW = wind_nw
        self.window_SE = wind_se

        #correcting logging values, if neccessary
        if self.logging_block_width > self.window_NW: self.logging_block_width = self.window_NW

    
    def drawIgnitionDay(self):
        #This function draws a random day to simulate an ignition. 
        #Possible return values are:
        #  -1 -> no ignition this year
        #  positive integer -> date (out of 365) of the ignition
        day = -1
        
        roll = random.uniform(0,1)
        
        if roll < self.ignition_prob:
            #this indicates that there is an ignition of importance
            # so pick a day
            day = random.randint(0,364)
        
        return day
    
    def drawLocation(self):
        #This function chooses a random location on the landscape and returns
        #  it as a two-element list
        
        #lets try it with fires anywhere:
        #setting a buffer of 2 makes sure that fuelave24 and timberave24 won't fail
        xloc = random.randint(2,self.width-2)
        yloc = random.randint(2,self.height-2)

        #original code with bug: Fires would only start in the central window
        #  i think it was intentional at first, but it was shortsighted
        if self.USE_BUGS:
            xloc = random.randint(self.window_NW, self.window_SE)
            yloc = random.randint(self.window_NW, self.window_SE)

        return [xloc,yloc]
    
    def tempMean(self, date):
        #This function returns the mean temperature on the given date
        #It is based on a cosine function which bottoms out on Day 0/365
        
        #Since a pure cosine starts at 1, then drops to -1 at pi, and back to
        # 1 at 2pi, I'm just going to flip it over by multiplying by -1, and then
        # adjusting the range so that entering 365 will be the same as 2pi, etc...
        
        radian_day = (2 * date * math.pi) / (365)
        
        cos = -1 * math.cos(radian_day)
        
        #secondly, the range has to be adjusted, such that -1 corresponds to 
        # self.temp_winter_low, and 1 corresponds to self.temp_summer_high
        
        cos = cos + 1 #yields a value from 0 to 2
        cos = cos / 2 #yields a value from 0 to 1
        
        #and finally, to scale it to the temperature extremes:
        tempmean = (self.temp_summer_high - self.temp_winter_low) * cos
        tempmean += self.temp_winter_low
        
        return tempmean
  
    def tempVar(self, date):
        #This function returns the variance (sigma) of the temperature distribution
        #   on the given date.
        #For the moment, this is just being held constant.
        return self.temp_var
        
    def drawTemperature(self, date):
        #This function takes a date (out of 365) and draws a day-time temperature.
        #Temperatures are drawn from bell-shaped distributions, whose parameters
        #  vary by date. For instance, the mean of the distribution will be higher
        #  in the mid-summer, and lower elsewhere, whereas the variance of the 
        #  distribution might be greater during the beginning and end of summer.
        
        #To pull from a normal distribution, the function call is:
        #random.normalvariate(mu, sigma)
        
        return random.normalvariate(self.tempMean(date), self.tempVar(date))
        
    def drawWindSpeed(self, date):
        #This function takes a date (out of 365) and draws a day-time wind speed.
        #Wind speeds are taken from an exponential distribution
        
        #note that I am not currently using the date argument which will mean that
        #  the wind distribution is the same for every date
        
        #The function signature is:
        #random.expovariate(lambd) where lambd should be (1 / mean)
        
        #return random.expovariate(1 / self.wind_mean) #this wasn't working, apparently?
        return (random.expovariate(1) * self.wind_mean)
    
    def drawEndOfFire(self):
        day = random.expovariate(1) * self.fire_average_end_day
        return day + 1  #give them at least one day...

    def generateNewLandscape(self):
        #This function will erase all current data in this pathway and 
        #  generate new values for timber/fuels, etc... It will also reset the 
        #  current year to 0.
        
        self.year = 0
        
        #invoking diamond-square algorithm to make new timber_value and fuel_load values
        #both timber_value and fuel_load have ranges between 0 and 100
        
        #function signature: 
        #def FireGirl_DS_alg      (seedValue,      min_val, max_val, roughness=0.5, USE_FUEL_BUG=False)
        newgrids = FireGirl_DS_alg(self.ID_number,       0,     120,           0.5, self.USE_BUGS)
        
        #assign them to this object's members
        #.timber_value = newgrids[0][:] #(mainly to size the grid properly)
        self.stand_age = newgrids[0]
        self.fuel_load = newgrids[1]
        
        #cast DS_algs floats to ints, for stand age
        for i in range(self.width):
            for j in range(self.height):
                #val = int(self.stand_age[i][j])
                self.stand_age[i][j] = int(self.stand_age[i][j])
        
        #assign timber values
        #for i in range(self.width):
        #    for j in range(self.height):
                #now assign growth values
                
        #        if self.using_growth_model == 1:
        #            self.timber_value[i][j] = self.growth_model_1[self.stand_age[i][j]]
        #        else:
        #            self.timber_value[i][j] = self.growth_model_2[self.stand_age[i][j]]
                
        
    def calcFuelAve8(self, xloc, yloc):
        #This function averages the fuel load of the 8 cells surrounding the one
        #  indicated by the input paramters
        
        total_fuel8 = 0
        for i in range(xloc-1, xloc+1):
            for j in range(yloc-1, yloc+1):
                #don't count the actual cell... just the adjacent 8 cells
                if not (i == xloc and j == yloc):
                    total_fuel8 += self.fuel_load[i][j]
                    
        fuel_ave8 = total_fuel8 / 8
        
        return fuel_ave8
        
    def calcTimberAve8(self, xloc, yloc):
        #This function averages the timber value of the 8 cells surrounding the one
        #  indicated by the input paramters
        
        total_timber8 = 0
        for i in range(xloc-1, xloc+1):
            for j in range(yloc-1, yloc+1):
                #don't count the actual cell... just the adjacent 8 cells
                if not (i == xloc and j == yloc):
                    total_timber8 += self.getPresentTimberValue(i,j) #self.timber_value[i][j]
                    
        timber_ave8 = total_timber8 / 8
        
        return timber_ave8
        
    def calcFuelAve24(self, xloc, yloc):
        #This function averages the fuel load of the 24 cells surrounding the
        # one indicated by the input arguments
        
        total_fuel24 = 0
        for i in range(xloc-2, xloc+2):
            for j in range(yloc-2, yloc+2):
                #don't count the actual cell... just the adjacent 24 cells
                if not (i == xloc and j == yloc):
                    total_fuel24 += self.fuel_load[i][j]
                    
        fuel_ave24 = total_fuel24 / 24    
        
        return fuel_ave24
        
    def calcTimberAve24(self, xloc, yloc):
        #This function averages the timber value of the 24 cells surrounding the
        # one indicated by the input arguments
        
        total_timber24 = 0
        for i in range(xloc-2, xloc+2):
            for j in range(yloc-2, yloc+2):
                #don't count the actual cell... just the adjacent 24 cells
                if not (i == xloc and j == yloc):
                    total_timber24 += self.getPresentTimberValue(i,j) #self.timber_value[i][j]
                    
        timber_ave24 = total_timber24 / 24
        
        return timber_ave24
    
    def evaluateSuppressionRule(self, ignite_date, ignite_loc, ignite_wind, ignite_temp):
        #This function passes the current state of things (ignition, features,
        #  etc...) to the Policy object, and gets back a suppression probability
        
        #The policy object functions being used have signatures:
        #def FireGirl_Policy.setValues(windspeed, temp, date, timber_val, timber_ave8, timber_ave24, fuel, fuel_ave8, fuel_ave24)
        #def FireGirl_Policy.evaluateSuppressionProbability():
        
        #values needed:
        timber_val = 0
        timber_ave8 = 0
        timber_ave24 = 0
        fuel = 0
        fuel_ave8 = 0
        fuel_ave24 = 0
        
        xloc = ignite_loc[0]
        yloc = ignite_loc[1]
        
        #the easy ones:
        #SPEED
        timber_val = self.getPresentTimberValue(xloc, yloc) #self.timber_value[xloc][yloc]
        fuel = self.fuel_load[xloc][yloc]
        
        #in the following four calculations, we don't have to worry about checking 
        #  bounds, because the ignition locs are only being pulled from the center 
        #  of the landscape, and there's a 43-cell buffer in every direction.
                
        #calculating averages from adjacent 8 cells
        timber_ave8 = self.calcTimberAve8(xloc, yloc)
        fuel_ave8 = self.calcFuelAve8(xloc, yloc)
            
        #calculating averages from adjacent 24 cells
        timber_ave24 = self.calcTimberAve24(xloc, yloc)
        fuel_ave24 = self.calcFuelAve24(xloc, yloc)
        
        #OLD POLICY TYPE
        #for reference, the Policy object function signature i'm using:
        #def self.setValues  (  windspeed,        temp,        date, timber_val, timber_ave8, timber_ave24, fuel, fuel_ave8, fuel_ave24)
        #self.Policy.setValues(ignite_wind, ignite_temp, ignite_date, timber_val, timber_ave8, timber_ave24, fuel, fuel_ave8, fuel_ave24)  
        #I'm assuming that the Policy object has already loaded the appropriate parameters
        #pol_val = self.Policy.evaluateSuppressionProbability()

        d2 = ignite_date * ignite_date
        features = [1, ignite_date, d2, ignite_temp, ignite_wind, timber_val, timber_ave8, timber_ave24, fuel, fuel_ave8, fuel_ave24]
        #self.Policy.setFeatures(features) #Un-needed
        pol_val = self.Policy.calcProb(features)

        if pol_val == None:
            print("Error: Invalid type for policy probability: type = None")

        return pol_val
        
    def calcFireSpreadRate(self, wind, temp, fuel):
        #This function calculates the logistic function that governs fire spread
        #   rates. The parameters below are arbitrary, and give rise to the
        #   shape I wanted for the model:
        
        out_scale = self.fire_param_outputscale
        in_scale = self.fire_param_inputscale
        zero_adj = self.fire_param_zeroadjust
        smooth = self.fire_param_smoothness
        
        exponent = (   -1 * smooth *     (   ((wind + temp + fuel) / in_scale) - zero_adj   )       )
        fspread = out_scale / (1 + math.exp(exponent))
        
        #Enforcing minimum spread restrictions
        if (wind + temp) < self.min_spread_windtemp:
            fspread = 0
        if fuel < self.min_spread_fuel:
            fspread = 0
        
        return fspread
    
    def calcCrownFireRisk(self, fuel):
        #This function takes the current fuel load in a cell and calculates the
        #   probabilty that a fire will spread to it's crown. It is based entirely
        #   on fuel_load. The shape of the logistic is arbitrary, and gives
        #   a crownfire model that should provide good tradeoffs between low-fuel
        #   fires and high-fuel fires.
        
        #Parameters that shape the logistic function:
        out_scale = self.crownfire_param_outputscale
        in_scale = self.crownfire_param_inputscale
        zero_adj = self.crownfire_param_zeroadjust
        smooth = self.crownfire_param_smoothness
        
        #Calculating the logistic
        exponent = (   -1 * smooth *     (   (fuel / in_scale) - zero_adj   )       )
        cf_risk = out_scale / (1 + math.exp(exponent))
        
        return cf_risk

    def getGrowth(self, year):
        """Returns the timber growth that happened in a specific year"""
        if year < len(self.yearly_growth_totals) and year >= 0:
            return self.yearly_growth_totals[year]
        else:
            return -1.0

    def getHarvest(self, year):
        """Returns the harvest total in the speficied year"""
        if year < len(self.yearly_logging_totals) and year >= 0:
            return self.yearly_logging_totals[year]
        else:
            return -1.0

    def getSuppressionCost(self,year):
        """Returns the suppression costs in the speficied year"""
        if year < len(self.yearly_suppression_costs) and year >= 0:
            return self.yearly_suppression_costs[year]
        else:
            return -1.0

    def getTimberLossTotal(self):
        #This function looks through all the fires that have occurred and adds up their
        # losses in timber values

        loss = 0
        #look through past ignitions and add up timber loss
        for ign in self.ignition_events:
            #the following line is used when the fire outcomes are recorded
            #firerecord_new.setOutcomes([timber_loss, cells_burned, sup_cost, end_time])
            outcomes = ign.getOutcomes()

            #subtract timber loss
            loss += outcomes[0]

        return round(loss, 0)

    def getTimberLossFrom(self, start_year, end_year):
        """Returns the harvest values between the specified years
        """
        #santatizing inputs
        if start_year < 0: start_year = 0
        if end_year > len(self.ignition_events): end_year = len(self.ignition_events)

        total = 0
        for i in range(start_year, end_year):
            outcomes = self.ignition_events[i].getOutcomes()
            total += outcomes[0]

        return round(total)

    def getHarvestTotal(self):
        """Returns the total harvest value to date that has occured in this pathway's history
        """

        total = 0

        for entry in self.yearly_logging_totals:
            total += entry

        return round(total, 0)

    def getHarvestFrom(self, start_year, end_year):
        """Returns the harvest values between the specified years
        """
        #santatizing inputs
        if start_year < 0: start_year = 0
        if end_year > len(self.yearly_logging_totals): end_year = len(self.yearly_logging_totals)

        total = 0
        for i in range(start_year, end_year):
            total += self.yearly_logging_totals[i]

        return round(total)

    def getSuppressionTotal(self):
        """Returns the total suppression cost to date in this pathway's history
        """

        total = 0

        for entry in self.yearly_suppression_costs:
            total += entry

        return round(total, 0)

    def getSuppressionFrom(self, start_year, end_year):
        """Returns the suppression costs between the specified years
        """
        #santatizing inputs
        if start_year < 0: start_year = 0
        if end_year > len(self.yearly_suppression_costs): end_year = len(self.yearly_suppression_costs)

        total = 0
        for i in range(start_year, end_year):
            total += self.yearly_suppression_costs[i]

        return round(total)

    def getGrowthTotal(self):
        """Returns the total timber growth to date in this pathway's history
        """

        total = 0

        for entry in self.yearly_growth_totals:
            total += entry

        return round(total, 0)

    def getGrowthFrom(self, start_year, end_year):
        """Returns the total growth between the specified years
        """
        #santatizing inputs
        if start_year < 0: start_year = 0
        if end_year > len(self.yearly_growth_totals): end_year = len(self.yearly_growth_totals)

        total = 0
        for i in range(start_year, end_year):
            total += self.yearly_growth_totals[i]

        return round(total)

    def updateNetValue(self):
        #This function makes the pathway go through it's full history and add/subtract up 
        #  all the costs and gains it's incurred in it's history. This value is then
        #  assigned to self.net_value, and also returned to the function caller.

        #reset net value
        self.net_value = 0

        #add/subtract up the various components
        self.net_value = self.getHarvestTotal() - self.getSuppressionTotal()# - self.getTimberLossTotal()

        #the value is already assigned to the local variable, but also return it:
        return self.net_value

    def recordCurrentStateToHistory(self):
        """Make value-copies of the current landscape grids and add them to the history lists
        """

        #making arrays to hold the copied information
        #fuel_copy = []
        #timber_copy = []

        #copying values
        # for i in range(len(self.fuel_load)):
        #    append a new column in each list
            # fuel_copy.append([])
            # timber_copy.append([])
            # for j in range(len(self.fuel_load[0])):
        #        add this element to the most recently added column
         #       using +1-1 to ensure copy-by-value instead of copy-by-reference
                # fuel_copy[i].append(self.fuel_load[i][j] + 1.0 - 1.0)
        #        timber_copy[i].append(self.timber_value[i][j] + 1.0 - 1.0)
                # timber_copy[i].apppend(self.getPresentTimberValue(i,j))
                
        #self.fuel_load_history.append(fuel_copy)
        #self.timber_value_history.append(timber_copy)
        
        pass

    def saveImage(self, filename=None, imagetype="composite", showburns=0):
        """Creates an image of the current landcape

        Arguments
        filename: the name of the image file to be saved. This can be a file-like object as well
        
        imagetype: the particular view of the landscape to be made
        options are:
        - "composite" which is the sum of fuel and timber
        - "fuel" which shows only the fuel values
        - "timber" which shows only the timber values
        
        showburns: an integer which indicates how may past burns (starting at the current year)
         to overlay on the image.

        Returns
        Nothing

        """

        #use default filename as necessary
        if filename == None:
            filename = "image_output_" + str(imagetype) + str(self.ID_number) + "_" + str(self.year) + ".bmp"


        image = Image.new("RGBA",(self.width,self.height))
        #choosing pixel color
        #my lowest value is green: (0,  240,0, 255)
        #my middle value is yellow:(240,240,0, 255)
        #my high value is red:     (240,  0,0, 255)

        #lowest value is 0
        #highest value is dependent on the type of image... 
        #  for composite, it's around 200
        #  for fuel or timber alone, it's around 100

        #so if the value is below half, we'll hold green = 240 and vary red from 0-240
        #   if the value is above half, we'll hold red = 240, and vary green from 240-0

        #setting high value for the image. This will associate with RED colors
        high_val = 100
        mid_val = 50
        if imagetype == "composite":
            high_val = 200
            mid_val = 100

        current_pixel = 0
        scaled_val = 0
        green = 0
        red = 0

        for i in range(self.width):
            for j in range(self.height):


                #check the current value
                if imagetype == "composite":
                    current_pixel = self.fuel_load[i][j] + self.getPresentTimberValue(i,j) #self.timber_value[i][j]

                elif imagetype == "fuel":
                    current_pixel = self.fuel_load[i][j]

                else: #imagetype == "timber":
                    current_pixel = self.getPresentTimberValue(i,j) #self.timber_value[i][j]


                #constraining excessively values (which are allowed in the simulator)
                if current_pixel > high_val:
                    current_pixel = high_val

                #which half are we on?
                if current_pixel < mid_val:
                    #we're on the low half of the scale
                    green = 240
                    #setting red:
                    #-current_pixel ranges from 0 - mid_val
                    #-a value of 0 means a 0% (out of 240) value for red
                    #-a value of mid_val means 100% (out of 240) value for red
                    #-a value of 1/2 * mid_val means 50% (out of 240) value for red
                    #etc...
                    red = (current_pixel / mid_val) * 240

                else:
                    #we're on the high half of the scale
                    red = 240
                    #setting green...
                    #-at mid_val, green starts at 240, and decreases linearly to 0
                    #  once it current_val = high_val (or more)
                    green = 240 * ((-1 * (current_pixel - high_val)) / mid_val)


                #assign values
                image.putpixel( (i,j), (int(red),int(green),0, 255) )

            #finished with this column, on to the next

        #finished with all pixels

        #now draw burns, if requested
        if showburns > 0:
            opacity = 255

            burn_image = Image.new("RGBA",(self.width,self.height))

            #look at the FireLog entries for each desired year
            for burn_year in range(self.year - showburns, self.year):
                if burn_year < 0: continue

                #choose the new opacity amount
                opacity = 255 - int( (255/showburns) * (self.year - burn_year) )

                # burn records have entries in the following format:
                #[time, location, spread_rate, crown_burned, "ignition"]   

                #look at each cell ignition in the current FireLog and draw that pixel
                for burn_event in self.FireLog[burn_year].burn_events:
                    loc = burn_event[1]
                    #check location veracity
                    if (loc[0] < 0) or (loc[1] < 0) or (loc[0] >= self.width) or (loc[1] >= self.height):
                        print("burn location out of bounds: " + str(loc))
                        continue
                    burn_image.putpixel( (loc[0], loc[1]) , (0,0,0,opacity) )

            #now blend the images
            image = Image.alpha_composite(image, burn_image)        


        image.save(filename, "PNG")


    def doFire(self, ignite_date, ignite_loc, ignite_wind, ignite_temp, suppress):
        """This function is the fire model. Given the input arguments, it will
        conduct the process of spreading a fire accross the landscape
        
        Steps:
           1) Determine how much "time" to simulate, based on
               a) suppress or let-burn
               b) severity of the weather
               c) local forest conditions
               d) some randomness (exponential?)
           2) Start the priority queue by calculating ignition times from the
               first cell to it's neigbors.
               a) as each cell burns, record the timber value lost
           3) Do the Priority queue until the alloted time is finished
        """
    
        xloc = ignite_loc[0]
        yloc = ignite_loc[1]
        reach = self.fire_param_reach
        
        #starting a new firelog to keep track of what happens.
        fire_log_item = FireGirlFireLog(self.year)
        
        
        end_time = self.drawEndOfFire()
        
        current_time = 0
        
        #construct the priority queue and add the first cell to it with time = 0
        pqueue = []
        pqueue.append([0,ignite_loc])
        
        #setting a variable that will hold the lowest of all the ingition times in the queue.
        next_ign = 1000
        
        #set up an array to mark which cells have already been burned over
        #SPEED
        #burned = [[False]*self.height]*self.width
        #for i in range(129):
        #    burned.append([])
        #    for j in range(129):
        #        burned[i].append(False)
        
        #set up an array to mark which cells have their crowns' burned
        #crown_burned = [[False]*self.height]*self.width
        #for i in range(129):
        #    crown_burned.append([])
        #    for j in range(129):
        #        crown_burned[i].append(False)

        burned = [None]*self.width
        crown_burned = [None]*self.width
        for i in range(self.height):
            burned[i] = [False]*self.height
            crown_burned[i] = [False]*self.height

                
        #start the queue loop
        iter_cap = 5000
        iter_count = 0
        while True:
            #failsafe exit
            iter_count += 1
            if iter_count > iter_cap:
                if not self.SILENT:
                    print("A very large fire has occured... Pathway: " + str(self.ID_number) + 
                          "  Year: " + str(self.year) + 
                          "  Burn Time: " + str( round(current_time,3) )    )
                break
                
            #check to make sure that there is at least one queued arrival
            if len(pqueue) == 0:
                #no queued arrivals, so there's no fire, so we're done
                #print("Priority Queue Exiting: No more queued ignitions")
                break
            
            #look through all the queued ignitions and find the earliest ignition
            #  time.
            next_ign = 10000
            for ign in pqueue:
                #if this queued arrival is earlier than the current best arrival,
                #  keep IT as the NEW earliest arrival
                if ign[0] < next_ign:
                    next_ign = ign[0]
                    #and update our current location
                    xloc = ign[1][0]
                    yloc = ign[1][1]
            
            #now check to see if the soonest arrival happens before the time is up.
            if next_ign >= end_time:
                #no fire arrivals (ignitions) are in queue within the alloted time
                #   so the firespread has stopped.
                #print("Priority Queue Exiting: Remaining queued ignitions are past the time limit")
                break
            
            #we haven't left the loop, so the next arrival is valid, so look at
            #  it and add its neighbors to the queue
            
            #moving current time up to this ignition
            current_time = next_ign
            
            #setting this cell to burned
            burned[xloc][yloc] = True
            
            
            #Calculating this cell's fire spreadrate, which needs it's fuel load, too
            fuel_ld = self.fuel_load[xloc][yloc] 
            spreadrate = self.calcFireSpreadRate(ignite_wind, ignite_temp, fuel_ld)

            #add the effects of suppression
            if suppress == True:
                spreadrate *= self.fire_suppression_rate 

            
            # Check if the crown will burn (if the spreadrate is > 0)
            # Timber loss is a probabalistic function based on the 
            #   calcCrownFireRisk() function.  This function will return
            #   a probability of crownfire, and we'll roll a uniform
            #   number against it.
            
            #TODO: FIX THIS HACK
            #because crownfire is stochaistic as originally implemented and is ALSO effected by policies:
            # it's changing the random number draw counts, and making otherwise identical pathways with
            # separate policies to have different weather. This isn't good. So I'm hacking in a deterministic
            # version of crownfirerisk for now...
            
            #original model call
            if self.USE_BUGS:
                roll = random.uniform(0,1)
                #SPEED
                if roll < self.calcCrownFireRisk(fuel_ld):
                    crown_burned[xloc][yloc] = True
            else:   
                #HACK
                hack_val = (fuel_ld * 2) + (spreadrate * 2)
                #SPEED
                if hack_val > self.getPresentTimberValue(xloc,yloc): # self.timber_value[xloc][yloc]:
                    crown_burned[xloc][yloc] = True
                    

            #if the fire spreadrate of this fire is 0, then don't bother checking
            #   for neighbors and calculating arrival times... there won't be any
            #   spread, and for that matter, we'll get a divide-by-zero error.
            if spreadrate == 0:
                #no spreadrate, so we can't calculate arrival times, etc...
                #remove this item from the queue, and move on.
                
                #print(".doFire() is removing an item from the queue because of Zero spreadrate")
                #print("   queue length:  before = " + str(len(pqueue)) ),
                pqueue.remove([current_time,[xloc,yloc]])
                #print("  after = " + str(len(pqueue)) )
                
                continue
            
            
            #recording information in the Logbook item
            #function signature is:  FireGirlfireLog.addIgnitionEvent(time, location, spread_rate, crown_burned):
            fire_log_item.addIgnitionEvent(current_time, [xloc,yloc], spreadrate, crown_burned[xloc][yloc]) 
            
            
            
            dist = 0
            arrival_time = 0
            
            
            #setting iteration ranges
            x_low = xloc-reach
            x_high = xloc+reach+1
            y_low = yloc-reach
            y_high = yloc+reach+1
            
            if self.USE_BUGS:
                x_high = xloc+reach
                y_high = yloc+reach

            #checking bounds
            if (x_low < 0): x_low = 0
            if (y_low < 0): y_low = 0
            if (x_high >= self.width): x_high = self.width - 1
            if (y_high >= self.height): y_high = self.height - 1
            
            
            for i in range(x_low, x_high):
                for j in range(y_low, y_high):

                    #don't calculate time to the current cell
                    if not ((xloc == i) and (yloc == j)):
                        
                        #we're checking each neighbor within the reach range, so
                        #  first, we need to check whether it's already been
                        #  burned over
                        
                        if burned[i][j] == False:
                            
                            #this neighbor hasn't burned over yet, so:
                            # 1) calculate a new time-till arrival
                            # 2) check to see if this neighbor is already in the queue
                            # 2a) if it is, then check to see if this arrival time is sooner
                            #       and if so, update it. Otherwise, just move on.
                            # 2b) if it isn't in the queue, then add it as a new queue item
                            
                            # 1) arrival time for this neighbor
                            dist = math.sqrt(   (xloc - i)*(xloc - i) + (yloc-j)*(yloc-j)    )
                            arrival_time = (dist/spreadrate) + current_time
                            
                            # 2) checking to see if this neighbor is already queued
                            found_in_q = False
                            for ign in pqueue:
                                if ign[1][0] == i and ign[1][1] == j:
                                    #this neighbor IS in the queue already, so check its arrival time
                                    #print("   neighbor found in queue... updating...")
                                    found_in_q = True
                                    
                                    #updating it's arrival time if need be
                                    if arrival_time < ign[0]:
                                        #the new arrival time is sooner, so update this queue item
                                        ign[0] = arrival_time

                                    
                            #check to see if we ever found this neighbor
                            if found_in_q == False:
                                #we never found it, so it wasn't in the queue, and it's not burned, so add it
                                pqueue.append([arrival_time, [i,j]])

            # we've now finished checking the neighbors, so it's time to remove this item from the queue
            pqueue.remove([current_time,[xloc,yloc]])
        
        
        # and now we've exited the priority queue as well   
        
        #look through the burned cells and update the actual grid values
        #Also, record losses
        timber_loss = 0
        cells_burned = 0
        cells_crowned = 0
        
        for i in range(self.width):
            for j in range(self.height):
                if burned[i][j] == True:
                    cells_burned += 1
                                        
                    #this cell was burned, so set the fuel_load to zero, and apply
                    #  the crown-burning model to the timber_value
                    self.fuel_load[i][j] = 0
                    
                    
                    
                    #adding up timber loss 
                    if crown_burned[i][j] == True:  #this was set when spreadrate was calculated earlier
                        #the crown burned, so record the loss and set it to zero
                        
                        #SPEED
                        ####BOTH Lines are modified for self.getPresentTimberValue(i,j)###########
                        timber_loss += self.getPresentTimberValue(i,j) #self.timber_value[i][j]
                        #self.timber_value[i][j] = 0
                        ####################
                        
                        cells_crowned += 1
                        
                        #and reset the age so that self.year + self.stand_age = 0
                        self.stand_age[i][j] = -1 * self.year
        
        
        
        #Adding the final results to the fire_log_item
        fire_log_item.updateResults(timber_loss, cells_burned, cells_crowned)

        #Adding the lists (maps) as well
        fire_log_item.map_burned = burned
        fire_log_item.map_crowned = crown_burned
        
        #add the FireLog item to the pathway's list (it's just an ordinary list)
        self.FireLog.append(fire_log_item)


        #add up suppression cost and record it
        sup_cost = 0
        if suppress == True:
            sup_cost +=  cells_burned * self.fire_suppression_cost_per_cell
            sup_cost +=      end_time * self.fire_suppression_cost_per_day

        self.yearly_suppression_costs.append(sup_cost)
        

        #and finally, return the loss data
        return [timber_loss, cells_burned, sup_cost, end_time, cells_crowned]

    
    def setGrowthModel(self, model_number):
        if model_number == 1:
            self.growth_model = self.growth_model_1
        elif model_number == 2:
            self.growth_model = self.growth_model_2
        else:
            pass

    def getPresentTimberValue(self,x,y):
        """ Uses the stand_age at the given coordinates to return the timber value given the current year
        """
        
        #timber_index = self.stand_age[x][y] + self.year

        #sanatizing, just in case
        #if timber_index > 299: timber_index = 299
        #if timber_index < 0: timber_index = 0
        
        #if self.using_growth_model == 1:
        #    return self.growth_model_1[timber_index]
        #else: #elif self.using_growth_model == 2:
        #    return self.growth_model_2[timber_index]

        return self.growth_model[self.stand_age[x][y] + self.year]

    def doGrowth(self):
        #This function applies the timber and fuel load growth models to all cells on the landscape,
        #  including those outside of the window of interest.

        #The current model is simply:  
        #       timber value next year = constant * ln(timber age next year)
        #
        # and to comput age this year, just invert:
        #       timber age this year = exp(timber value this year / constant)
        #
        # so first find the current age, then add one, and then find next year's timber value


        #declaring loop variables
        age = 0
        old_val = 0
        new_val = 0

        #to record total growth this year (by use in the logging model, etc...)
        total_growth = 0.0

        #loop over every cell
        for i in range(self.width):
            for j in range(self.height):


                ##########################################################
                # TIME INTENSIVE METHOD
                #For summation purposes, record current value
                #old_val = self.timber_value[i][j]
                
                # 1) Apply timber growth equation:
                # #calculate current age
                # age = math.exp(self.timber_value[i][j] / self.growth_timber_constant)
                # #and apply the timber value for the new age
                # old_val = self.timber_value[i][j]
                # new_val = self.growth_timber_constant * math.log(age + 1)
                # self.timber_value[i][j] = new_val
                ###########################################################
                             
                
                #NOTE: to save simulation time, stand ages are not incremented each year.
                #  Instead, the current year is incremented and just added to the last age recorded.
                #  When a fire burns or timber is cut, the stand age is set to -1 * self.year, so that
                ##  when you add self.year to self.stand_age[x][y], you get 0, which is the new ACTUAL stand age.
                
                old_val = self.growth_model[self.stand_age[i][j] + self.year]
                new_val = self.growth_model[self.stand_age[i][j] + self.year + 1]
                    
                #and record this growth as part of the year's total growth, but only for
                #  the window of interest
                #TODO: update to fix timber harvest in the whole landscape
                if self.USE_BUGS:
                    if (i >= self.window_NW) and (i < self.window_SE) and (j >= self.window_NW) and (j < self.window_SE):
                        total_growth += (new_val - old_val)
                else:
                    total_growth += (new_val - old_val)

                ######################################
                # 2) Apply fuel accumulation model:  #
                ######################################
                self.fuel_load[i][j] += self.growth_fuel_accumulation


        #record this year's yearly growth total
        self.yearly_growth_totals.append(total_growth)

    def doLogging_one_block(self, x, y, timber_limit):
        #this function will cut stands in the block given, starting at (x,y), and using
        #   self.logging_block_width to determine the block size. It will cut stands
        #   until it has met or exceeded timber_limit in cut value. When it has reached
        #   the end of the block, or met the timber_limit quota, it will stop, and return
        #   the total value cut.

        total_cut = 0
        done_cutting = False

        #start loop at (x,y) and move over the block defined by self.logging_block_width, etc...
        for i in range(x, x + self.logging_block_width):
            
            #check for stop condition
            if done_cutting == True: break

            for j in range(y, y + self.logging_block_width):

                #check for stop condition
                if done_cutting == True: break

                #check if this stand is at least the minimum value
                #if self.timber_value[i][j] >= self.logging_min_value:
                if self.getPresentTimberValue(i,j) >= self.logging_min_value:
                    
                    ########BOTH lines have been changed!!!######
                    #this stand is eligible to be cut, so do it, and record the total
                    total_cut += self.getPresentTimberValue(i,j) #self.timber_value[i][j]
                    #self.timber_value[i][j] = 0
                    ############################################
                    
                    #and reset the stand age so that self.year + stand_age[i][j] = 0
                    self.stand_age[i][j] = -1 * self.year

                    #and "burn the slash"
                    self.fuel_load[i][j] = self.logging_slash_remaining

                    #check if we've hit our limit
                    if total_cut >= timber_limit:
                        #we have, so break out of the for loops
                        done_cutting = True

        #check for failure
        #if done_cutting == False:
        #    print("doLogging_one_block() failed to cut quota")

        #finished looping, so return the total amoutn cut
        return total_cut

    def doLogging(self):
        #This function starts cutting timber in randomly selected blocks within the window
        #  of interest until the self-imposed logging limit has been reached. The logging limit
        #  is set to a percentage of the year's accumulated growth, as calculated by the 
        #  growth model.  The model will not cut stands under a certain value. For instance,
        #  at an age of 30, stands have a value of about 75, so potentially no stands under a
        #  value of 75 (or whatever is set by the user in the class constructor) will be harvested
        #  in the current block
        #
        #If the current block does not have enough harvestable timber to meet the quota, then
        #  whatever remains of the quota will be looked for in a new harvest block (which may
        #  or may not overlap the current one; it's random), and so on.


        #get this year's maximum allowed cut: It will be the last entry in the tree growth 
        #  list
        max_cut = self.yearly_growth_totals[len(self.yearly_growth_totals)-1]

        #reduce maximum according to logging model parameters
        max_cut = max_cut * self.logging_percentOfIncrement

        # Cut stands in this block until
        #   a) enough timber has been cut for the year
        #   b) there are no more eligible stands
        total_cut = 0

        tries = 0

        #generate enough random numbers for the whole cuttng sequence. This avoids different
        # numbers of random number draws based on timber availability
        x_locs = []
        y_locs = []
        if self.USE_BUGS:
            for i in range(self.logging_max_cuts):
                x_locs.append(random.randint(self.window_NW, self.window_SE - self.logging_block_width ))
                y_locs.append(random.randint(self.window_NW, self.window_SE - self.logging_block_width ))
        else:
            for i in range(self.logging_max_cuts):
                x_locs.append(random.randint(0, self.width  - self.logging_block_width ))
                y_locs.append(random.randint(0, self.height - self.logging_block_width ))


        # while True:
        #     if total_cut >= max_cut: break
        #     if tries >= self.logging_max_cuts: break

        #     # Select a logging block such that the whole block will fit within the window
        #     #     of interest
        #     #TODO: Fix this so that logging happens everywhere
        #     x = random.randint(43, 86-self.logging_block_width )
        #     y = random.randint(43, 86-self.logging_block_width )

        #     #cut a new block, allowing whatever remains of our current logging limit to be cut
        #     total_cut += self.doLogging_one_block( x, y, (max_cut - total_cut) )
        #     #print("while loop, total_cut = " + str(total_cut))
        #     #print("max cut = " + str(max_cut))

        #     tries += 1

        for i in range(self.logging_max_cuts):
            if total_cut >= max_cut: break

            #cut a new block, allowing whatever remains of our current logging limit to be cut
            total_cut += self.doLogging_one_block( x_locs[i], y_locs[i], (max_cut - total_cut) )


        # Report logging results
        self.yearly_logging_totals.append(total_cut)
        


    def doOneYear(self):
        #This function advances the current pathway by one year
        #Steps:
        #   1) Draw a new ignition (if any)
        #   2) If there's an ignition, apply the current suppression rule
        #   3) If there's an ignition, do the fire model, including suppression (if any)
        #   4) Do the growth model
        #   5) Do the logging model
        #
        #   in each step) In this pathway's Logbook, record:
        #       a) the year
        #       b) the features of the ignition (if any)
        #       c) for reporting, fire results (timber loss and cells burned)
        #       d) the suppression decision and probability (if any)
        #       e) logging totals
        #       f) ecological variables???

        
        ##################################
        ### STEP 0 - RECORDING HISTORY ###
        ##################################

        #take the current fuel_load and timber_value lists and add them to the historical lists
        if self.SAVE_HISTORY:
            self.recordCurrentStateToHistory()
        
        
        ###############################
        ### STEP 1 - IGNITION MODEL ###
        ###############################
        
        ignite_date = self.drawIgnitionDay()
        ignite_loc = self.drawLocation()
        
        #for easier code-writing in a moment...
        x = ignite_loc[0]
        y = ignite_loc[1]
        
        #For recording in the new ignition object type
        firerecord_new = FireGirlIgnitionRecord()
        firerecord_new.year = self.year
        firerecord_new.location = [x,y]

        ignite_wind = 0
        ignite_temp = 0
        if ignite_date >= 0:
            ignite_wind = self.drawWindSpeed(ignite_date)
            ignite_temp = self.drawTemperature(ignite_date)
            
       
            #recording in the new ignition object type
            #SPEED
            features = [1, ignite_date, (ignite_date*ignite_date), ignite_temp, ignite_wind,
                        self.getPresentTimberValue(x,y), self.calcTimberAve8(x,y), self.calcTimberAve24(x,y),
                        self.fuel_load[x][y], self.calcFuelAve8(x,y), self.calcFuelAve24(x,y)]
            f_labels = ["CONSTANT", "date", "date squared", "temperature", "wind speed", 
                        "timber value", "timber value, ave8", "timber value, ave24", 
                        "fuel load", "fuel load, ave8", "fuel load, ave 24" ]
            firerecord_new.setFeatures(features)
            firerecord_new.setFeatureLabels(f_labels)

        
        
        ##################################
        ### STEP 2 - SUPPRESSION MODEL ###
        ##################################
        
        # setting some default values
        suppress_prob = 0
        suppress_decision = False
        
        if ignite_date == -1:
            #there is no important ignition this year
            pass
        else:
            #there is an important ignition this year
            #Apply the Suppression Rule
            suppress_prob = self.evaluateSuppressionRule(ignite_date, ignite_loc, ignite_wind, ignite_temp)
            suppress_decision = self.chooseSuppression(suppress_prob)
            
            #recording suppression data in the new logbook type
            firerecord_new.setChoice(suppress_decision)
            firerecord_new.setProb(suppress_prob)
        
        
        
        ###########################
        ### STEP 3 - FIRE MODEL ###
        ###########################
        if ignite_date == -1:
            #there is no important ignition this year
            pass
        else:
            #there is an ignition, so:
            
            # Invoke the fire model and record it's return value, which is in the
            # form:  [timber_loss, cells_burned, sup_cost, end_time, cells_crowned]
            fireresults = self.doFire(ignite_date, ignite_loc, ignite_wind, ignite_temp, suppress_decision)
            timber_loss = fireresults[0]
            cells_burned = fireresults[1]
            sup_cost = fireresults[2]
            end_time = round(fireresults[3], 3)
            cells_crowned = fireresults[4]
            
            #recording outcomes in new object type
            firerecord_new.setOutcomes([timber_loss, cells_burned, sup_cost, end_time, cells_crowned])
            firerecord_new.setOutcomeLabels(["Timber Loss", "Cells Burned", "Suppression Cost", "Burn Time", "Cells Crowned"])
        
                
        #################################
        ### STEP 4 -   GROWTH MODEL   ###
        #################################
    
        self.doGrowth()


        ##############################
        ### STEP 5 - LOGGING MODEL ###
        ##############################
        
        self.doLogging()

        
        ##########################
        ### Finalization Steps ###
        ##########################

        #adding new ignition record object to the appropriate list.
        #but first check for empty records
        if not firerecord_new.getFeatures() == []:  #this will happen if there was ever a no-ignition event.
            #there is a record, so append it.
            self.ignition_events.append(firerecord_new)
        else:
            print("no ignition???")

            print (firerecord_new.feature_labels) 
            print (firerecord_new.features)
            print (firerecord_new.outcome_labels) 
            print (firerecord_new.outcomes) 
            print("prob: "),
            print (firerecord_new.policy_prob)  
            print("choice: "),
            print (firerecord_new.policy_choice)  
            print("year: "),
            print (firerecord_new.year) 
            print("burn time: "),
            print (firerecord_new.burn_time)


        #Finally, advance the year by one.
        #print("Finishing Year " + str(self.year))
        self.year += 1
        #print("Beginning year " + str(self.year))


        #and update net values
        self.updateNetValue()

        
    def doYears(self, number_of_years):
        #this function will process any number of pathway years, starting with
        #  the current year in self.year.  After each, it will increment self.year
        #  by one and continue
        for y in range(number_of_years):
            self.doOneYear()
            #self.year += 1
            
    
    