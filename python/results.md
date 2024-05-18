## v0
**v0**
openning dataset: 0.019634439959190786s
gridding: 232.06691551499534s
fourier transform: 0.39494385302532464s

real    3m54.856s
user    3m30.872s
sys     0m23.525s


**notes:** 1 core of epyc

## v1
**v1**
openning dataset: 0.020160239073447883s
gridding: 76.40704975405242s
fourier transform: 0.39501506509259343s

real    1m19.187s
user    1m10.612s
sys     0m8.617s

## v2
**v2**
openning dataset: 0.019283825997263193s
gridding: 1.7558171639684588s
fourier transform: 0.3912698660278693s

real    0m4.732s
user    0m3.744s
sys     0m1.349s

**notes:** 1 core of epyc

## v3
**v3**
openning dataset: 0.18852494699240196s
gridding: 23.638135811997927s
fourier transform: 0.3929694109974662s

real    0m26.396s
user    0m50.379s
sys     0m13.156s


## v4
**v4**
openning dataset: 0.20819946100527886s
gridding: 26.77406486599648s
fourier transform: 0.39936628300347365s

real    0m29.628s
user    0m52.845s
sys     0m13.364s


## v5
**v6**
openning dataset: 0.2125791080034105s
gridding: 2.0906538140116027s
fourier transform: 0.3965039550093934s

real    0m4.844s
user    0m4.819s
sys     0m4.036s


## v6
[saliei@epyc007 python]$ time ./v6.py
**v6**
n_workers: 8
openning dataset: 0.20832469398737885s
gridding: 1.9457385369896656s
fourier transform: 0.39672539799357764s

real    0m4.644s
user    0m6.009s
sys     0m4.195s
[saliei@epyc007 python]$ vmi v6.py
bash: vmi: command not found
[saliei@epyc007 python]$ vim v6.py
[saliei@epyc007 python]$ time ./v6.py
**v6**
n_workers: 4
openning dataset: 0.18604543100809678s
gridding: 1.5404350349999731s
fourier transform: 0.3889356419967953s

real    0m4.228s
user    0m5.372s
sys     0m3.849s
[saliei@epyc007 python]$ vim v6.py
[saliei@epyc007 python]$ time ./v6.py
**v6**
n_workers: 16
openning dataset: 0.18632192400400527s
gridding: 2.990788358001737s
fourier transform: 0.3987353639968205s

real    0m5.693s
user    0m8.174s
sys     0m4.832s
[saliei@epyc007 python]$ vim v6.py
[saliei@epyc007 python]$ vim v6.py
[saliei@epyc007 python]$ time ./v6.py
**v6**
n_workers: 32
openning dataset: 0.20795436899061315s
gridding: 5.300075590988854s
fourier transform: 0.398038721003104s

real    0m8.013s
user    0m11.890s
sys     0m6.169s
[saliei@epyc007 python]$ vim v6.py
[saliei@epyc007 python]$ time ./v6.py
**v6**
n_workers: 64
openning dataset: 0.1848883369966643s
gridding: 9.817494039991288s
fourier transform: 0.4032062530022813s

real    0m12.580s
user    0m18.960s
sys     0m9.019s

