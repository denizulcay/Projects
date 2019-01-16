import numpy as np

means = np.loadtxt('../predictions/means.csv')
stdev = np.loadtxt('../predictions/std.csv')

stitches = np.load('../predictions/1-stitches.npz')
end2ends = np.load('../predictions/2-end2end-with-stitches.npz')

stitches_ = []
end2ends_ = []
for i in range(len(stitches.files)):
    s = stitches['arr_'+str(i)]
    e = end2ends['arr_'+str(i)]

    if s.std() != 0:
        s = means[i] + (s - s.mean()) * (stdev[i] / s.std())
    if e.std() != 0:
        e = means[i] + (e - e.mean()) * (stdev[i] / e.std())

    stitches_.append(s)
    end2ends_.append(e)

np.savez('../predictions/3-stitches-scaled.npz', *stitches_)
np.savez('../predictions/4-end2end-with-stitches-scaled.npz', *end2ends_)
