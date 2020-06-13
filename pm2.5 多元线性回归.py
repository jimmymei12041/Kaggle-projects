



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  data preparation
pd.set_option('Display.max_rows', 5000)
pd.set_option('Display.max_columns', 40)
pd.set_option('Display.width', 200000)

raw_data = pd.read_csv('train.csv', encoding='GB18030')

raw_data[raw_data == 'NR'] = 0

data = raw_data.iloc[0:, 3:]
print(type(data))


data = data.to_numpy().astype(float)
print(type(data))
print(data.shape)



y_all = []
y_last_all = []
PM10_all = []

for i in range(0, 240):
    y = data[i*18+9, 9]
    y_last = data[i*18+9, 8]
    PM10 = data[i*18+8, 8]
    y_all.append(y)
    y_last_all.append(y_last)
    PM10_all.append(PM10)


# correlation visualization
def heatmap():
    y_all = []
    y_last_all = []
    ABM_all = []
    CH4_all = []
    CO_all = []
    NMHC_all = []
    NO_all = []
    NO2_all = []
    NOx_all = []
    O3_all = []
    PM10_all = []
    RH_all = []
    SO2_all = []
    THC_all = []
    WD_HR_all = []
    WIND_DIREC_all = []
    WIND_SPEED_all = []
    WS_HR_all = []

    for i in range(0, 240):
        y = data[i*18+9, 9]
        y_last = data[i*18+9, 8]
        ABM = data[i*18, 8]
        CH4 =  data[i*18+1, 8]
        CO = data[i*18+2, 8]
        NMHC = data[i*18+3, 8]
        NO = data[i*18+4, 8]
        NO2 = data[i*18+5, 8]
        NOx = data[i*18+6, 8]
        O3 = data[i*18+7, 8]
        PM10 = data[i*18+8, 8]
        RH = data[i*18+11, 8]
        SO2 = data[i*18+12, 8]
        THC= data[i*18+13, 8]
        WD_HR= data[i*18+14, 8]
        WIND_DIREC= data[i*18+15, 8]
        WIND_SPEED= data[i*18+16, 8]
        WS_HR= data[i*18+17, 8]

        y_all.append(y)
        y_last_all.append(y_last)
        ABM_all.append(ABM)
        CH4_all.append(CH4)
        CO_all.append(CO)
        NMHC_all.append(NMHC)
        NO_all.append(NO)
        NO2_all.append(NO2)
        NOx_all.append(NOx)
        O3_all.append(O3)
        PM10_all.append(PM10)
        RH_all.append(RH)
        SO2_all.append(SO2)
        THC_all.append(THC)
        WD_HR_all.append(WD_HR)
        WIND_DIREC_all.append(WIND_DIREC)
        WIND_SPEED_all.append(WIND_SPEED)
        WS_HR_all.append(WS_HR)


    y_all = pd.DataFrame(y_all)
    y_last_all = pd.DataFrame(y_last_all)
    ABM_all = pd.DataFrame(ABM_all)
    NMHC_all = pd.DataFrame(NMHC_all)
    NO_all = pd.DataFrame(NO_all)
    NO2_all = pd.DataFrame(NO2_all)
    NOx_all = pd.DataFrame(NOx_all)
    O3_all = pd.DataFrame(O3_all)
    PM10_all = pd.DataFrame(PM10_all)
    RH_all = pd.DataFrame(RH_all)
    SO2_all = pd.DataFrame(SO2_all)
    THC_all = pd.DataFrame(THC_all)
    WD_HR_all = pd.DataFrame(WD_HR_all)
    WIND_DIREC_all = pd.DataFrame(WIND_DIREC_all)
    WIND_SPEED_all = pd.DataFrame(WIND_SPEED_all)
    WS_HR_all = pd.DataFrame(WS_HR_all)

    df_close = pd.concat([y_all, y_last_all, ABM_all, NMHC_all, NO_all,NO2_all,NOx_all,O3_all,PM10_all,RH_all,SO2_all,THC_all,WD_HR_all,WIND_DIREC_all,WIND_SPEED_all,WS_HR_all], axis=1)

    f = plt.figure(figsize=(13, 12))
    plt.matshow(df_close.corr(), fignum=f.number)
    plt.xticks(np.arange(16),['PM2.5','PM2.5 -1','ABM','NMHC','NO','NO2','NOx','O3','PM10','RH','SO2','THC','WD','WIND_DIREC','WIND_SPEED','WS_HR'])
    plt.yticks(np.arange(16),['PM2.5','PM2.5 -1','ABM','NMHC','NO','NO2','NOx','O3','PM10','RH','SO2','THC','WD','WIND_DIREC','WIND_SPEED','WS_HR'])
    cb = plt.colorbar()
    #cb.ax.tick_params(labelsize=9)
    plt.title('Correlation Matrix', fontsize=10)

    return plt.show()

heat = heatmap()
print(heat)

print(y_all)
print(y_last_all)


y_all = np.array(y_all)
y_last_all = np.array(y_last_all)



# basic setups
theta1 = 1
theta2 = 1
theta0 = 0
d_SSR_theta1 = 0
d_SSR_theta2 = 0
d_SSR_theta0 = 0
ii = 0
learn = 0.0007

# Training, find theta1 and theta0
while ii < 30000:
    for i in range(len(y_all)):
        d_SSR_theta1 += (theta1 * (y_last_all[i]) + theta2 * (PM10_all[i]) + theta0 - y_all[i]) * y_last_all[i]
        d_SSR_theta2 += (theta1 * (y_last_all[i]) + theta2 * (PM10_all[i]) + theta0 - y_all[i]) * PM10_all[i]
        d_SSR_theta0 += (theta1 * (y_last_all[i]) + theta2 * (PM10_all[i]) + theta0 - y_all[i])

    MSE1 = d_SSR_theta1/len(y_all)
    MSE2 = d_SSR_theta2/len(y_all)
    MSE0 = d_SSR_theta0/len(y_all)
    theta1 = theta1 - learn * MSE1
    theta2 = theta2 - learn * MSE2
    theta0 = theta0 - learn * MSE0
    d_SSR_theta1 = 0
    d_SSR_theta2 = 0
    d_SSR_theta0 = 0

    ii += 1



print(theta1)
print(theta2)
print(theta0)



func = lambda func_x: func_x*theta1 + theta0
listx = []
listy = []
listx.append(-3)
listx.append(100)

listy.append(func(-3))
listy.append(func(100))

print(listx)
print(listy)

plt.scatter(y_last_all, y_all)
plt.plot(listx, listy, color = 'r')
plt.show()




test_data = pd.read_csv('test.csv')

test_data = test_data.iloc[:, 2:]
test_data = test_data.to_numpy()
aaa = np.empty([1,9])
test_data = np.insert(test_data, 0, values=aaa, axis=0)
print(test_data)
print(test_data.shape)

#  extract testing data
test1_all = []
test2_all = []
for i in range(0, 4320, 18):
    test1_all.append(test_data[10+i, 8])
    test2_all.append(test_data[9+i, 8])


print(test1_all[0])
print(test2_all[0])
final_output = np.empty([240, 1])


for i in range(240):
    final_output[i] = (int(test1_all[i])*theta1 + int(test2_all[i])*theta2 + theta0)

print(final_output)


# load to a new csv file
'''import csv
with open('submit2.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), final_output[i][0]]
        csv_writer.writerow(row)
        print(row)'''





