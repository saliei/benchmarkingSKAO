## v1

[saliei@epyc006 C_python]$ export OMP_NUM_THREADS=128
[saliei@epyc006 C_python]$ ./v1.py
gridding time: 0.04805187
fft time: 0.40122838



[saliei@epyc006 C_python]$ export OMP_NUM_THREADS=64
[saliei@epyc006 C_python]$ ./v1.py
gridding time: 0.03728498
fft time: 0.39777504


[saliei@epyc006 C_python]$ export OMP_NUM_THREADS=32
[saliei@epyc006 C_python]$ ./v1.py
gridding time: 0.03479346
fft time: 0.39423534
[saliei@epyc006 C_python]$ ./v1.py
gridding time: 0.03450636
fft time: 0.39370722


[saliei@epyc006 C_python]$ export OMP_NUM_THREADS=16
[saliei@epyc006 C_python]$ ./v1.py
gridding time: 0.03586636
fft time: 0.39740865


[saliei@epyc006 C_python]$ export OMP_NUM_THREADS=8
[saliei@epyc006 C_python]$ ./v1.py
gridding time: 0.05354183
fft time: 0.39723877
[saliei@epyc006 C_python]$ ./v1.py
gridding time: 0.05412535
fft time: 0.39699304
[saliei@epyc006 C_python]$ export OMP_NUM_THREADS=4
[saliei@epyc006 C_python]$ ./v1.py
gridding time: 0.09428858
fft time: 0.39044639
[saliei@epyc006 C_python]$ export OMP_NUM_THREADS=2
[saliei@epyc006 C_python]$ ./v1.py
gridding time: 0.17200444
fft time: 0.39880775
[saliei@epyc006 C_python]$ export OMP_NUM_THREADS=1
[saliei@epyc006 C_python]$ ./v1.py
gridding time: 0.31663389
fft time: 0.39809119


-----
## v2

--ntasks=32 --cpus-per-task=4

-np 1:
rank: 0, gridding time: 0.34004848
rank: 0, fft time:      0.39134481

-np 2:
rank: 1, gridding time: 0.19332126
rank: 0, gridding time: 0.20364447
rank: 0, fft time:      0.38999175

-np 4:
rank: 3, gridding time: 0.05734871
rank: 2, gridding time: 0.08093865
rank: 1, gridding time: 0.10281370
rank: 0, gridding time: 0.12836761
rank: 0, fft time:      0.39030823

-np 8:
rank: 3, gridding time: 0.05594776
rank: 5, gridding time: 0.05128000
rank: 1, gridding time: 0.04944843
rank: 7, gridding time: 0.04574171
rank: 2, gridding time: 0.09218352
rank: 6, gridding time: 0.10384614
rank: 4, gridding time: 0.15012697
rank: 0, gridding time: 0.17004829
rank: 0, fft time:      0.39054523


-np 16:
rank: 7, gridding time: 0.04049205
rank: 13, gridding time: 0.04318959
rank: 5, gridding time: 0.04543538
rank: 11, gridding time: 0.04515694
rank: 3, gridding time: 0.04605841
rank: 9, gridding time: 0.05920478
rank: 15, gridding time: 0.04281143
rank: 1, gridding time: 0.06423552
rank: 6, gridding time: 0.12121675
rank: 2, gridding time: 0.11532692
rank: 10, gridding time: 0.11378447
rank: 14, gridding time: 0.13486349
rank: 4, gridding time: 0.17494999
rank: 12, gridding time: 0.18224726
rank: 8, gridding time: 0.18380855
rank: 0, gridding time: 0.18575368
rank: 0, fft time:      0.39192826


-np 32:
rank: 5, gridding time: 0.06675992
rank: 13, gridding time: 0.06017232
rank: 31, gridding time: 0.06274045
rank: 27, gridding time: 0.06548513
rank: 21, gridding time: 0.06434331
rank: 23, gridding time: 0.06772643
rank: 29, gridding time: 0.06274912
rank: 15, gridding time: 0.06668748
rank: 7, gridding time: 0.06531320
rank: 17, gridding time: 0.06565823
rank: 11, gridding time: 0.06627587
rank: 19, gridding time: 0.06417781
rank: 3, gridding time: 0.06579751
rank: 9, gridding time: 0.06985114
rank: 1, gridding time: 0.06388358
rank: 25, gridding time: 0.07079799
rank: 30, gridding time: 0.12974500
rank: 18, gridding time: 0.14410633
rank: 22, gridding time: 0.11835307
rank: 26, gridding time: 0.11996104
rank: 2, gridding time: 0.12124537
rank: 14, gridding time: 0.12096519
rank: 6, gridding time: 0.12120936
rank: 10, gridding time: 0.12568073
rank: 20, gridding time: 0.19368139
rank: 28, gridding time: 0.19798495
rank: 4, gridding time: 0.19891125
rank: 12, gridding time: 0.19929911
rank: 24, gridding time: 0.23721619
rank: 8, gridding time: 0.25101742
rank: 16, gridding time: 0.26491694
rank: 0, gridding time: 0.27440751
rank: 0, fft time:      0.39526808


--ntasks=128 --cpus-per-task=1

-np 1:
rank: 0, gridding time: 0.34089238
rank: 0, fft time:      0.38847538

-np 2:
rank: 1, gridding time: 0.19327325
rank: 0, gridding time: 0.19855311
rank: 0, fft time:      0.40330604

-np 4:
rank: 3, gridding time: 0.11238725
rank: 2, gridding time: 0.13494528
rank: 1, gridding time: 0.15981393
rank: 0, gridding time: 0.18145166
rank: 0, fft time:      0.39151395

-np 8:
rank: 1, gridding time: 0.07871435
rank: 7, gridding time: 0.08722782
rank: 5, gridding time: 0.07617452
rank: 3, gridding time: 0.09315687
rank: 6, gridding time: 0.12024363
rank: 2, gridding time: 0.14110973
rank: 4, gridding time: 0.23035373
rank: 0, gridding time: 0.25574931
rank: 0, fft time:      0.38978854


-np 16:
rank: 5, gridding time: 0.09555610
rank: 13, gridding time: 0.08908125
rank: 15, gridding time: 0.07187197
rank: 11, gridding time: 0.05799902
rank: 3, gridding time: 0.06130434
rank: 1, gridding time: 0.11583973
rank: 7, gridding time: 0.09082931
rank: 9, gridding time: 0.10149958
rank: 14, gridding time: 0.10449146
rank: 10, gridding time: 0.11002586
rank: 2, gridding time: 0.10646786
rank: 6, gridding time: 0.10736498
rank: 12, gridding time: 0.16915528
rank: 4, gridding time: 0.18584794
rank: 8, gridding time: 0.19491605
rank: 0, gridding time: 0.19825032
rank: 0, fft time:      0.40106561

-np 32:
rank: 3, gridding time: 0.04729954
rank: 5, gridding time: 0.21859649
rank: 1, gridding time: 0.24843445
rank: 13, gridding time: 0.05946914
rank: 27, gridding time: 0.04664298
rank: 11, gridding time: 0.10862577
rank: 19, gridding time: 0.06680284
rank: 2, gridding time: 0.23928980
rank: 29, gridding time: 0.15441658
rank: 7, gridding time: 0.13117077
rank: 21, gridding time: 0.08712956
rank: 25, gridding time: 0.08411782
rank: 9, gridding time: 0.07963420
rank: 17, gridding time: 0.07652038
rank: 15, gridding time: 0.09920285
rank: 31, gridding time: 0.07594855
rank: 23, gridding time: 0.10068150
rank: 26, gridding time: 0.17928251
rank: 10, gridding time: 0.11044218
rank: 6, gridding time: 0.10094932
rank: 18, gridding time: 0.11270222
rank: 14, gridding time: 0.09947202
rank: 30, gridding time: 0.10071502
rank: 22, gridding time: 0.10185326
rank: 4, gridding time: 0.22885202
rank: 12, gridding time: 0.20208618
rank: 28, gridding time: 0.18449769
rank: 20, gridding time: 0.18421226
rank: 8, gridding time: 0.20334959
rank: 24, gridding time: 0.21246231
rank: 16, gridding time: 0.22943949
rank: 0, gridding time: 0.30619041
rank: 0, fft time:      0.39549401

-np 64:
rank: 3, gridding time: 0.03668362
rank: 13, gridding time: 0.04844039
rank: 51, gridding time: 0.11619676
rank: 59, gridding time: 0.06931609
rank: 45, gridding time: 0.05597580
rank: 61, gridding time: 0.06467769
rank: 35, gridding time: 0.09518047
rank: 53, gridding time: 0.05424860
rank: 19, gridding time: 0.05157295
rank: 43, gridding time: 0.05878628
rank: 27, gridding time: 0.05877526
rank: 11, gridding time: 0.05872703
rank: 41, gridding time: 0.06889712
rank: 47, gridding time: 0.06265214
rank: 21, gridding time: 0.10197025
rank: 29, gridding time: 0.25348240
rank: 37, gridding time: 0.09649127
rank: 5, gridding time: 0.51481741
rank: 23, gridding time: 0.06237469
rank: 25, gridding time: 0.08419446
rank: 33, gridding time: 0.06536967
rank: 63, gridding time: 0.19425704
rank: 49, gridding time: 0.06530057
rank: 31, gridding time: 0.10460237
rank: 55, gridding time: 0.07942393
rank: 15, gridding time: 0.06910060
rank: 1, gridding time: 0.10120132
rank: 7, gridding time: 0.47031748
rank: 17, gridding time: 0.06816186
rank: 39, gridding time: 0.13566326
rank: 9, gridding time: 0.39150430
rank: 57, gridding time: 0.12302079
rank: 26, gridding time: 0.23190797
rank: 46, gridding time: 0.21489859
rank: 42, gridding time: 0.14282048
rank: 34, gridding time: 0.15140478
rank: 50, gridding time: 0.19913336
rank: 18, gridding time: 0.35543921
rank: 22, gridding time: 0.21977883
rank: 10, gridding time: 0.35347064
rank: 62, gridding time: 0.13280128
rank: 58, gridding time: 0.18512846
rank: 30, gridding time: 0.13521769
rank: 14, gridding time: 0.13385392
rank: 54, gridding time: 0.13847334
rank: 6, gridding time: 0.14558322
rank: 38, gridding time: 0.13654222
rank: 20, gridding time: 0.19985257
rank: 44, gridding time: 0.33947370
rank: 2, gridding time: 0.57626950
rank: 28, gridding time: 0.21800441
rank: 52, gridding time: 0.36521600
rank: 36, gridding time: 0.24472546
rank: 12, gridding time: 0.45336863
rank: 60, gridding time: 0.30361306
rank: 24, gridding time: 0.27785589
rank: 40, gridding time: 0.36099509
rank: 4, gridding time: 0.29283179
rank: 56, gridding time: 0.26150497
rank: 48, gridding time: 0.32405387
rank: 8, gridding time: 0.30142198
rank: 16, gridding time: 0.34705731
rank: 32, gridding time: 0.46053235
rank: 0, gridding time: 0.38682933
rank: 0, fft time:      0.40710213

-np 128:
rank: 89, gridding time: 0.31891446
rank: 9, gridding time: 0.08400032
rank: 3, gridding time: 0.06269934
rank: 99, gridding time: 0.45168206
rank: 41, gridding time: 0.08775547
rank: 19, gridding time: 0.04961621
rank: 115, gridding time: 0.44208884
rank: 51, gridding time: 0.05769911
rank: 127, gridding time: 0.06280123
rank: 5, gridding time: 0.14729218
rank: 105, gridding time: 0.06670367
rank: 21, gridding time: 0.64332597
rank: 45, gridding time: 1.22609896
rank: 85, gridding time: 0.11162479
rank: 49, gridding time: 0.06683266
rank: 27, gridding time: 0.06704180
rank: 87, gridding time: 0.07423194
rank: 35, gridding time: 0.07956395
rank: 43, gridding time: 1.20372250
rank: 109, gridding time: 0.09121097
rank: 61, gridding time: 0.08548920
rank: 73, gridding time: 0.13944062
rank: 117, gridding time: 1.34285569
rank: 69, gridding time: 0.09908829
rank: 103, gridding time: 0.18418836
rank: 57, gridding time: 0.12028508
rank: 93, gridding time: 0.08835609
rank: 53, gridding time: 0.09195246
rank: 79, gridding time: 0.09484305
rank: 65, gridding time: 0.12379311
rank: 121, gridding time: 1.08656087
rank: 50, gridding time: 0.27102126
rank: 83, gridding time: 0.15639794
rank: 67, gridding time: 0.12230263
rank: 13, gridding time: 0.08288948
rank: 107, gridding time: 0.23676166
rank: 31, gridding time: 0.08342474
rank: 77, gridding time: 0.21048661
rank: 81, gridding time: 0.10718444
rank: 7, gridding time: 0.08365027
rank: 25, gridding time: 0.12974540
rank: 39, gridding time: 0.09649089
rank: 71, gridding time: 0.07966347
rank: 101, gridding time: 0.24371481
rank: 1, gridding time: 0.75695072
rank: 123, gridding time: 0.49877693
rank: 125, gridding time: 0.16099373
rank: 113, gridding time: 0.19861346
rank: 97, gridding time: 0.16964058
rank: 17, gridding time: 0.22582801
rank: 95, gridding time: 0.11850118
rank: 119, gridding time: 0.08319183
rank: 55, gridding time: 0.08731327
rank: 111, gridding time: 0.07974567
rank: 59, gridding time: 0.22556232
rank: 63, gridding time: 0.08150845
rank: 15, gridding time: 0.09674120
rank: 33, gridding time: 0.23101743
rank: 47, gridding time: 0.07784571
rank: 86, gridding time: 0.24966403
rank: 29, gridding time: 0.11217862
rank: 23, gridding time: 0.08958755
rank: 37, gridding time: 0.16891390
rank: 91, gridding time: 0.19052761
rank: 42, gridding time: 0.15921606
rank: 75, gridding time: 0.69775119
rank: 11, gridding time: 0.48145426
rank: 102, gridding time: 0.16575226
rank: 78, gridding time: 0.16154769
rank: 126, gridding time: 0.24989609
rank: 26, gridding time: 0.23117468
rank: 114, gridding time: 0.28193992
rank: 82, gridding time: 0.16931925
rank: 98, gridding time: 0.34431027
rank: 18, gridding time: 0.45753700
rank: 66, gridding time: 0.17517137
rank: 106, gridding time: 0.17899984
rank: 34, gridding time: 0.21222580
rank: 30, gridding time: 0.18384936
rank: 58, gridding time: 0.16780734
rank: 122, gridding time: 0.18477404
rank: 6, gridding time: 1.46273505
rank: 74, gridding time: 0.13427354
rank: 70, gridding time: 0.21981992
rank: 10, gridding time: 0.13559493
rank: 118, gridding time: 0.21749521
rank: 90, gridding time: 0.15892423
rank: 54, gridding time: 0.24386177
rank: 110, gridding time: 0.22419509
rank: 38, gridding time: 0.26704209
rank: 62, gridding time: 0.36296332
rank: 94, gridding time: 0.21668250
rank: 46, gridding time: 0.23044596
rank: 14, gridding time: 0.24636088
rank: 22, gridding time: 0.30083805
rank: 2, gridding time: 0.47614813
rank: 84, gridding time: 0.33249238
rank: 76, gridding time: 0.32254899
rank: 124, gridding time: 0.30921189
rank: 68, gridding time: 0.46015746
rank: 100, gridding time: 0.35235833
rank: 52, gridding time: 0.49898088
rank: 108, gridding time: 0.79987259
rank: 28, gridding time: 0.34648954
rank: 36, gridding time: 0.31991354
rank: 116, gridding time: 0.40938667
rank: 20, gridding time: 0.45472854
rank: 92, gridding time: 0.84084161
rank: 4, gridding time: 0.45576409
rank: 60, gridding time: 0.43416066
rank: 44, gridding time: 0.48287749
rank: 12, gridding time: 0.43696010
rank: 72, gridding time: 0.48635472
rank: 120, gridding time: 0.50426752
rank: 104, gridding time: 0.59389954
rank: 56, gridding time: 0.53365551
rank: 40, gridding time: 0.71175085
rank: 88, gridding time: 1.24916800
rank: 24, gridding time: 0.52818213
rank: 8, gridding time: 0.75866608
rank: 48, gridding time: 1.17976142
rank: 112, gridding time: 0.55481026
rank: 80, gridding time: 0.57806781
rank: 16, gridding time: 0.59070484
rank: 96, gridding time: 0.59700268
rank: 32, gridding time: 0.59054580
rank: 64, gridding time: 0.68994221
rank: 0, gridding time: 0.67451002
rank: 0, fft time:      0.43646528

---
## v3

export OMP_NUM_THREADS=1
gridding time: 0.16222181
fft time: 0.39176541

export OMP_NUM_THREADS=2
gridding time: 0.08876267
fft time: 0.39064849

export OMP_NUM_THREADS=4
gridding time: 0.04572602
fft time: 0.39071379

export OMP_NUM_THREADS=8
gridding time: 0.03015849
fft time: 0.39206691


export OMP_NUM_THREADS=16
gridding time: 0.02676671
fft time: 0.39244940


export OMP_NUM_THREADS=32
gridding time: 0.03026888
fft time: 0.39907465


export OMP_NUM_THREADS=64
gridding time: 0.03473859
fft time: 0.39374324

export OMP_NUM_THREADS=128
gridding time: 0.04532792
fft time: 0.39529701

---
## v4
