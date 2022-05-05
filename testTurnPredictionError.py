
cloud = []
snow = []
rain = []
og = []

with open("output_original.txt", "r") as txt_file:
    og = txt_file.readlines()
with open("output_cloud.txt", "r") as txt_file:
    cloud = txt_file.readlines()
with open("output_snow.txt", "r") as txt_file:
    snow = txt_file.readlines()
with open("output_rain.txt", "r") as txt_file:
    rain = txt_file.readlines()

c=0
for i in range(len(cloud)):
    if (og[i]!=rain[i]):
        # print(cloud[i],rain[i])
        c+=1
print("rain - error ",c/len(rain))        

c=0
for i in range(len(cloud)):
    if (og[i]!=snow[i]):
        # print(cloud[i],snow[i])
        c+=1
print("snow - error ",c/len(snow)) 

c=0
for i in range(len(cloud)):
    if (cloud[i]!=og[i]):
        # print(cloud[i],snow[i])
        c+=1
print("cloud - error ",c/len(cloud)) 
