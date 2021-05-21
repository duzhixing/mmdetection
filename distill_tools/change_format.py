
strs = "40.2 60.7 43.8 22.7 44.0 53.2 53.9 33.5 57.6 68.8"
s = strs.split()
and_s = "&".join(s)
col_s = "|".join(s)
print(and_s)
print(col_s)
a = map(lambda x:round(float(x) / 100,3),s)
print(list(a))
