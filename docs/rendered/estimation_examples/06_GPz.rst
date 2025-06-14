GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.3470328e-01	 3.1812014e-01	-3.2495787e-01	 3.2982647e-01	[-3.4830043e-01]	 4.6749830e-01


.. parsed-literal::

       2	-2.6373974e-01	 3.0701358e-01	-2.3940577e-01	 3.1953388e-01	[-2.7869853e-01]	 2.3355961e-01


.. parsed-literal::

       3	-2.2161813e-01	 2.8798373e-01	-1.8091269e-01	 2.9857921e-01	[-2.2875782e-01]	 2.9757357e-01
       4	-1.8302280e-01	 2.6325239e-01	-1.4123406e-01	 2.7301622e-01	[-2.0409583e-01]	 1.8736768e-01


.. parsed-literal::

       5	-9.7770986e-02	 2.5465949e-01	-6.2818319e-02	 2.6474511e-01	[-1.1577003e-01]	 1.9814348e-01


.. parsed-literal::

       6	-6.1882984e-02	 2.4920033e-01	-3.0282247e-02	 2.5955964e-01	[-7.0893114e-02]	 2.1130252e-01


.. parsed-literal::

       7	-4.5359022e-02	 2.4679469e-01	-2.0050950e-02	 2.5701789e-01	[-6.2197232e-02]	 2.2530651e-01


.. parsed-literal::

       8	-3.1761368e-02	 2.4460922e-01	-1.0863664e-02	 2.5462940e-01	[-5.3078654e-02]	 2.0918751e-01


.. parsed-literal::

       9	-1.7921759e-02	 2.4213155e-01	-1.8643974e-04	 2.5203219e-01	[-4.2820633e-02]	 2.1013570e-01


.. parsed-literal::

      10	-6.6974078e-03	 2.4020757e-01	 8.5741317e-03	 2.5129046e-01	[-3.9819868e-02]	 2.1014118e-01
      11	-1.8299560e-03	 2.3942557e-01	 1.2410312e-02	 2.5073271e-01	[-3.6405042e-02]	 1.8519592e-01


.. parsed-literal::

      12	 6.4781869e-04	 2.3887285e-01	 1.4729898e-02	 2.5002622e-01	[-3.3525678e-02]	 2.2715783e-01
      13	 4.6307387e-03	 2.3809143e-01	 1.8649182e-02	 2.4892551e-01	[-2.8222759e-02]	 1.8684411e-01


.. parsed-literal::

      14	 1.3830917e-01	 2.2290667e-01	 1.6172404e-01	 2.3068111e-01	[ 1.1935507e-01]	 4.4017196e-01


.. parsed-literal::

      15	 1.9728402e-01	 2.1992015e-01	 2.2358951e-01	 2.2907922e-01	[ 1.9301549e-01]	 3.2928872e-01


.. parsed-literal::

      16	 2.4746998e-01	 2.1458231e-01	 2.8448640e-01	 2.2324620e-01	[ 2.1531955e-01]	 2.0447731e-01


.. parsed-literal::

      17	 3.0879936e-01	 2.0986081e-01	 3.4277073e-01	 2.1764668e-01	[ 2.8315761e-01]	 2.0803928e-01


.. parsed-literal::

      18	 3.4489177e-01	 2.0561286e-01	 3.7746729e-01	 2.1359117e-01	[ 3.3136098e-01]	 2.0737481e-01
      19	 3.6969826e-01	 2.0251844e-01	 4.0236280e-01	 2.0952798e-01	[ 3.6330052e-01]	 2.0525217e-01


.. parsed-literal::

      20	 4.7243507e-01	 1.9894432e-01	 5.0525927e-01	 2.0379716e-01	[ 4.7124491e-01]	 2.2172046e-01
      21	 5.8734156e-01	 1.9946307e-01	 6.2485119e-01	 2.0501808e-01	[ 5.9740572e-01]	 1.9240522e-01


.. parsed-literal::

      22	 6.4855389e-01	 1.9264822e-01	 6.8562532e-01	 1.9762882e-01	[ 6.7082901e-01]	 2.0750976e-01
      23	 6.8497157e-01	 1.9212045e-01	 7.2259064e-01	 1.9777920e-01	[ 6.9974320e-01]	 1.8656445e-01


.. parsed-literal::

      24	 7.1829451e-01	 2.0066966e-01	 7.5544180e-01	 2.0785420e-01	[ 7.3064899e-01]	 1.9851089e-01


.. parsed-literal::

      25	 7.4936415e-01	 2.0237277e-01	 7.8730260e-01	 2.1023283e-01	[ 7.5835014e-01]	 2.2069550e-01
      26	 7.7326700e-01	 2.0008473e-01	 8.1219211e-01	 2.0889507e-01	[ 7.7931548e-01]	 1.9102716e-01


.. parsed-literal::

      27	 8.0532129e-01	 1.9969649e-01	 8.4481129e-01	 2.0972685e-01	[ 8.0707701e-01]	 2.1162915e-01


.. parsed-literal::

      28	 8.3657872e-01	 1.9745742e-01	 8.7674772e-01	 2.0580225e-01	[ 8.4192194e-01]	 2.1658206e-01
      29	 8.6774292e-01	 1.9284759e-01	 9.0885435e-01	 1.9957082e-01	[ 8.7587100e-01]	 1.9217968e-01


.. parsed-literal::

      30	 8.9928498e-01	 1.8943625e-01	 9.4160485e-01	 1.9433210e-01	[ 9.0924785e-01]	 2.2527361e-01


.. parsed-literal::

      31	 9.2899703e-01	 1.8535635e-01	 9.7245844e-01	 1.9002215e-01	[ 9.3841712e-01]	 2.0656443e-01


.. parsed-literal::

      32	 9.5806544e-01	 1.8114603e-01	 1.0027080e+00	 1.8542423e-01	[ 9.6106799e-01]	 2.1940589e-01


.. parsed-literal::

      33	 9.7991099e-01	 1.7732696e-01	 1.0242841e+00	 1.8022242e-01	[ 9.9259555e-01]	 2.0452023e-01


.. parsed-literal::

      34	 9.8520756e-01	 1.7547922e-01	 1.0315035e+00	 1.7811752e-01	  9.8540483e-01 	 2.2076058e-01
      35	 1.0084681e+00	 1.7345838e-01	 1.0539785e+00	 1.7568358e-01	[ 1.0109283e+00]	 2.0084143e-01


.. parsed-literal::

      36	 1.0172609e+00	 1.7218475e-01	 1.0628804e+00	 1.7401494e-01	[ 1.0194515e+00]	 2.0137525e-01


.. parsed-literal::

      37	 1.0327675e+00	 1.6959813e-01	 1.0784778e+00	 1.7104778e-01	[ 1.0345534e+00]	 2.1193099e-01


.. parsed-literal::

      38	 1.0454274e+00	 1.6611059e-01	 1.0915191e+00	 1.6685095e-01	[ 1.0412911e+00]	 2.0874047e-01
      39	 1.0610976e+00	 1.6380722e-01	 1.1070940e+00	 1.6451657e-01	[ 1.0574595e+00]	 1.9706607e-01


.. parsed-literal::

      40	 1.0726408e+00	 1.6172079e-01	 1.1187074e+00	 1.6260968e-01	[ 1.0676135e+00]	 2.1158528e-01


.. parsed-literal::

      41	 1.0884861e+00	 1.5891904e-01	 1.1348472e+00	 1.5949972e-01	[ 1.0835999e+00]	 2.2057080e-01


.. parsed-literal::

      42	 1.1098509e+00	 1.5444616e-01	 1.1566042e+00	 1.5405411e-01	[ 1.1098121e+00]	 2.1379018e-01


.. parsed-literal::

      43	 1.1251143e+00	 1.5390295e-01	 1.1721029e+00	 1.5303208e-01	[ 1.1270565e+00]	 2.1117759e-01


.. parsed-literal::

      44	 1.1373575e+00	 1.5271282e-01	 1.1845584e+00	 1.5170675e-01	[ 1.1361786e+00]	 2.1148849e-01


.. parsed-literal::

      45	 1.1507257e+00	 1.5035094e-01	 1.1985527e+00	 1.4856690e-01	[ 1.1445863e+00]	 2.1718979e-01


.. parsed-literal::

      46	 1.1645476e+00	 1.4821879e-01	 1.2124515e+00	 1.4542264e-01	[ 1.1605697e+00]	 2.1146464e-01
      47	 1.1793853e+00	 1.4661383e-01	 1.2275005e+00	 1.4297529e-01	[ 1.1691834e+00]	 2.0614934e-01


.. parsed-literal::

      48	 1.1884774e+00	 1.4517535e-01	 1.2372284e+00	 1.4056670e-01	[ 1.1747297e+00]	 1.9996977e-01


.. parsed-literal::

      49	 1.1993290e+00	 1.4387465e-01	 1.2479254e+00	 1.3986926e-01	[ 1.1835636e+00]	 2.0954347e-01


.. parsed-literal::

      50	 1.2088360e+00	 1.4300222e-01	 1.2575672e+00	 1.3903762e-01	[ 1.1963989e+00]	 2.2114158e-01


.. parsed-literal::

      51	 1.2190850e+00	 1.4173223e-01	 1.2680581e+00	 1.3782578e-01	[ 1.2097486e+00]	 2.1649504e-01
      52	 1.2270885e+00	 1.4032614e-01	 1.2761795e+00	 1.3636613e-01	[ 1.2240444e+00]	 1.8994856e-01


.. parsed-literal::

      53	 1.2356166e+00	 1.3981828e-01	 1.2847995e+00	 1.3625475e-01	[ 1.2312854e+00]	 2.0859694e-01


.. parsed-literal::

      54	 1.2445374e+00	 1.3900616e-01	 1.2939436e+00	 1.3578955e-01	[ 1.2361609e+00]	 2.2145391e-01


.. parsed-literal::

      55	 1.2534386e+00	 1.3825422e-01	 1.3035745e+00	 1.3481081e-01	[ 1.2398852e+00]	 2.1066856e-01


.. parsed-literal::

      56	 1.2614495e+00	 1.3741899e-01	 1.3118106e+00	 1.3371534e-01	[ 1.2450359e+00]	 2.1858311e-01
      57	 1.2658798e+00	 1.3723799e-01	 1.3161457e+00	 1.3334437e-01	[ 1.2519345e+00]	 2.0015788e-01


.. parsed-literal::

      58	 1.2715889e+00	 1.3679970e-01	 1.3218060e+00	 1.3246338e-01	[ 1.2590449e+00]	 2.1716094e-01


.. parsed-literal::

      59	 1.2776245e+00	 1.3621083e-01	 1.3279058e+00	 1.3137147e-01	[ 1.2638631e+00]	 2.2193503e-01


.. parsed-literal::

      60	 1.2847697e+00	 1.3561553e-01	 1.3350814e+00	 1.3030549e-01	[ 1.2684264e+00]	 2.1497154e-01


.. parsed-literal::

      61	 1.2922946e+00	 1.3517518e-01	 1.3426679e+00	 1.2961088e-01	[ 1.2707249e+00]	 2.1416163e-01


.. parsed-literal::

      62	 1.3006551e+00	 1.3466483e-01	 1.3512188e+00	 1.2896542e-01	[ 1.2748982e+00]	 2.1001506e-01


.. parsed-literal::

      63	 1.3091135e+00	 1.3454258e-01	 1.3599353e+00	 1.2886115e-01	[ 1.2810817e+00]	 2.0992017e-01


.. parsed-literal::

      64	 1.3155030e+00	 1.3427348e-01	 1.3663318e+00	 1.2835110e-01	[ 1.2864580e+00]	 2.0312357e-01


.. parsed-literal::

      65	 1.3199129e+00	 1.3394761e-01	 1.3706345e+00	 1.2794136e-01	[ 1.2921173e+00]	 2.1575093e-01


.. parsed-literal::

      66	 1.3263988e+00	 1.3347007e-01	 1.3772698e+00	 1.2721358e-01	[ 1.3005577e+00]	 2.2239637e-01


.. parsed-literal::

      67	 1.3326952e+00	 1.3338948e-01	 1.3836945e+00	 1.2675831e-01	[ 1.3079404e+00]	 2.2242451e-01


.. parsed-literal::

      68	 1.3375625e+00	 1.3310078e-01	 1.3884788e+00	 1.2649559e-01	[ 1.3130549e+00]	 2.0211744e-01


.. parsed-literal::

      69	 1.3427843e+00	 1.3293208e-01	 1.3937673e+00	 1.2633650e-01	[ 1.3174312e+00]	 2.0423770e-01


.. parsed-literal::

      70	 1.3468709e+00	 1.3260284e-01	 1.3981024e+00	 1.2620217e-01	[ 1.3185795e+00]	 2.2041082e-01


.. parsed-literal::

      71	 1.3523826e+00	 1.3235709e-01	 1.4037749e+00	 1.2596443e-01	[ 1.3218422e+00]	 2.0834398e-01


.. parsed-literal::

      72	 1.3585420e+00	 1.3200485e-01	 1.4101059e+00	 1.2550519e-01	[ 1.3240199e+00]	 2.1409941e-01


.. parsed-literal::

      73	 1.3636303e+00	 1.3172571e-01	 1.4153835e+00	 1.2477908e-01	[ 1.3279125e+00]	 2.2167301e-01
      74	 1.3691686e+00	 1.3129627e-01	 1.4209006e+00	 1.2411802e-01	[ 1.3309538e+00]	 1.9519830e-01


.. parsed-literal::

      75	 1.3744256e+00	 1.3095351e-01	 1.4262795e+00	 1.2323469e-01	[ 1.3332487e+00]	 2.1673918e-01


.. parsed-literal::

      76	 1.3783221e+00	 1.3078915e-01	 1.4303226e+00	 1.2284781e-01	[ 1.3373000e+00]	 2.1636152e-01


.. parsed-literal::

      77	 1.3821022e+00	 1.3076326e-01	 1.4341090e+00	 1.2255653e-01	[ 1.3418788e+00]	 2.2067237e-01


.. parsed-literal::

      78	 1.3868941e+00	 1.3075516e-01	 1.4389983e+00	 1.2228936e-01	[ 1.3472962e+00]	 2.1573305e-01


.. parsed-literal::

      79	 1.3908637e+00	 1.3073603e-01	 1.4430428e+00	 1.2187903e-01	[ 1.3508981e+00]	 2.2048521e-01


.. parsed-literal::

      80	 1.3956467e+00	 1.3080706e-01	 1.4478787e+00	 1.2173271e-01	[ 1.3526564e+00]	 2.1964669e-01


.. parsed-literal::

      81	 1.3988089e+00	 1.3057798e-01	 1.4510583e+00	 1.2151160e-01	  1.3517485e+00 	 2.2180104e-01
      82	 1.4012978e+00	 1.3042401e-01	 1.4534807e+00	 1.2132289e-01	  1.3513340e+00 	 1.8562102e-01


.. parsed-literal::

      83	 1.4034663e+00	 1.3014460e-01	 1.4556035e+00	 1.2119011e-01	  1.3509802e+00 	 2.2820640e-01
      84	 1.4075127e+00	 1.2987815e-01	 1.4597861e+00	 1.2078993e-01	  1.3501869e+00 	 1.9601679e-01


.. parsed-literal::

      85	 1.4110163e+00	 1.2936937e-01	 1.4634995e+00	 1.2040413e-01	  1.3478982e+00 	 2.1235371e-01


.. parsed-literal::

      86	 1.4138889e+00	 1.2950032e-01	 1.4663475e+00	 1.2028917e-01	  1.3522214e+00 	 2.1996021e-01


.. parsed-literal::

      87	 1.4167371e+00	 1.2962130e-01	 1.4692016e+00	 1.2010590e-01	[ 1.3579528e+00]	 2.1788406e-01


.. parsed-literal::

      88	 1.4194597e+00	 1.2967388e-01	 1.4719570e+00	 1.1998834e-01	[ 1.3623133e+00]	 2.1725154e-01


.. parsed-literal::

      89	 1.4216660e+00	 1.2967773e-01	 1.4743009e+00	 1.1971398e-01	[ 1.3646754e+00]	 2.1269345e-01


.. parsed-literal::

      90	 1.4249383e+00	 1.2958112e-01	 1.4774677e+00	 1.1958481e-01	[ 1.3694584e+00]	 2.1992350e-01


.. parsed-literal::

      91	 1.4261528e+00	 1.2943651e-01	 1.4786632e+00	 1.1950849e-01	  1.3682063e+00 	 2.1897626e-01


.. parsed-literal::

      92	 1.4286781e+00	 1.2919795e-01	 1.4812586e+00	 1.1917448e-01	  1.3666744e+00 	 2.1109509e-01
      93	 1.4326977e+00	 1.2889018e-01	 1.4854323e+00	 1.1857202e-01	  1.3656285e+00 	 1.9308424e-01


.. parsed-literal::

      94	 1.4365358e+00	 1.2870899e-01	 1.4894898e+00	 1.1778120e-01	  1.3692699e+00 	 2.0963788e-01
      95	 1.4385765e+00	 1.2869972e-01	 1.4915688e+00	 1.1760024e-01	[ 1.3744320e+00]	 1.7611361e-01


.. parsed-literal::

      96	 1.4406451e+00	 1.2861630e-01	 1.4935428e+00	 1.1760237e-01	[ 1.3791939e+00]	 1.9673109e-01
      97	 1.4420856e+00	 1.2859660e-01	 1.4949563e+00	 1.1758033e-01	[ 1.3826770e+00]	 1.8125367e-01


.. parsed-literal::

      98	 1.4444858e+00	 1.2844983e-01	 1.4973963e+00	 1.1756735e-01	[ 1.3854597e+00]	 2.2757959e-01


.. parsed-literal::

      99	 1.4464917e+00	 1.2849397e-01	 1.4996122e+00	 1.1764918e-01	[ 1.3857674e+00]	 2.1106505e-01


.. parsed-literal::

     100	 1.4491517e+00	 1.2822342e-01	 1.5022700e+00	 1.1771181e-01	  1.3834389e+00 	 2.1568441e-01


.. parsed-literal::

     101	 1.4507404e+00	 1.2809280e-01	 1.5039066e+00	 1.1761346e-01	  1.3805970e+00 	 2.0527530e-01
     102	 1.4528763e+00	 1.2789434e-01	 1.5061604e+00	 1.1743230e-01	  1.3750376e+00 	 1.7854857e-01


.. parsed-literal::

     103	 1.4553569e+00	 1.2762469e-01	 1.5087580e+00	 1.1711153e-01	  1.3720753e+00 	 2.0437932e-01


.. parsed-literal::

     104	 1.4574906e+00	 1.2729424e-01	 1.5110336e+00	 1.1659324e-01	  1.3644573e+00 	 2.0911145e-01


.. parsed-literal::

     105	 1.4597008e+00	 1.2722445e-01	 1.5131168e+00	 1.1654651e-01	  1.3715055e+00 	 2.1565604e-01


.. parsed-literal::

     106	 1.4608852e+00	 1.2718311e-01	 1.5142582e+00	 1.1656364e-01	  1.3737998e+00 	 2.0857668e-01
     107	 1.4623206e+00	 1.2715245e-01	 1.5156780e+00	 1.1639895e-01	  1.3784677e+00 	 1.8739915e-01


.. parsed-literal::

     108	 1.4639801e+00	 1.2691161e-01	 1.5174559e+00	 1.1622388e-01	  1.3757951e+00 	 2.0328903e-01


.. parsed-literal::

     109	 1.4657485e+00	 1.2692730e-01	 1.5193012e+00	 1.1613980e-01	  1.3763776e+00 	 2.0422864e-01
     110	 1.4672447e+00	 1.2675481e-01	 1.5209062e+00	 1.1598073e-01	  1.3748730e+00 	 1.8291998e-01


.. parsed-literal::

     111	 1.4688729e+00	 1.2658865e-01	 1.5226136e+00	 1.1594275e-01	  1.3751657e+00 	 2.1431541e-01
     112	 1.4706104e+00	 1.2618472e-01	 1.5243834e+00	 1.1584898e-01	  1.3774115e+00 	 1.7066479e-01


.. parsed-literal::

     113	 1.4720192e+00	 1.2603538e-01	 1.5257410e+00	 1.1584946e-01	  1.3813313e+00 	 1.9224453e-01


.. parsed-literal::

     114	 1.4742650e+00	 1.2576852e-01	 1.5279455e+00	 1.1572096e-01	[ 1.3870105e+00]	 2.1160460e-01


.. parsed-literal::

     115	 1.4751639e+00	 1.2564946e-01	 1.5289627e+00	 1.1530070e-01	[ 1.3971195e+00]	 2.1722007e-01


.. parsed-literal::

     116	 1.4771219e+00	 1.2559086e-01	 1.5308632e+00	 1.1519961e-01	  1.3952529e+00 	 2.0777059e-01


.. parsed-literal::

     117	 1.4782497e+00	 1.2552032e-01	 1.5320577e+00	 1.1496140e-01	  1.3945577e+00 	 2.1211243e-01


.. parsed-literal::

     118	 1.4792663e+00	 1.2545768e-01	 1.5331658e+00	 1.1472730e-01	  1.3955259e+00 	 2.1051145e-01


.. parsed-literal::

     119	 1.4811177e+00	 1.2534181e-01	 1.5351174e+00	 1.1450119e-01	[ 1.3989531e+00]	 2.0416522e-01


.. parsed-literal::

     120	 1.4825118e+00	 1.2528721e-01	 1.5365503e+00	 1.1438677e-01	[ 1.4042141e+00]	 3.3335400e-01


.. parsed-literal::

     121	 1.4841170e+00	 1.2520951e-01	 1.5381435e+00	 1.1437702e-01	[ 1.4082932e+00]	 2.0877051e-01
     122	 1.4853281e+00	 1.2515006e-01	 1.5392733e+00	 1.1442382e-01	[ 1.4111249e+00]	 1.9523335e-01


.. parsed-literal::

     123	 1.4864747e+00	 1.2514793e-01	 1.5403427e+00	 1.1459257e-01	[ 1.4128804e+00]	 2.1578503e-01


.. parsed-literal::

     124	 1.4875875e+00	 1.2494178e-01	 1.5413941e+00	 1.1456521e-01	  1.4121291e+00 	 2.1224141e-01


.. parsed-literal::

     125	 1.4884719e+00	 1.2484376e-01	 1.5422867e+00	 1.1448376e-01	[ 1.4130528e+00]	 2.0857096e-01
     126	 1.4898743e+00	 1.2457405e-01	 1.5437516e+00	 1.1445119e-01	  1.4119543e+00 	 1.7973185e-01


.. parsed-literal::

     127	 1.4913039e+00	 1.2432296e-01	 1.5452798e+00	 1.1460358e-01	  1.4106518e+00 	 2.2020340e-01


.. parsed-literal::

     128	 1.4927157e+00	 1.2408943e-01	 1.5467496e+00	 1.1481920e-01	  1.4094226e+00 	 2.3653650e-01


.. parsed-literal::

     129	 1.4937505e+00	 1.2397592e-01	 1.5478158e+00	 1.1496785e-01	  1.4071522e+00 	 2.2077727e-01


.. parsed-literal::

     130	 1.4948652e+00	 1.2386817e-01	 1.5489843e+00	 1.1503083e-01	  1.4050832e+00 	 2.1067762e-01
     131	 1.4955360e+00	 1.2380277e-01	 1.5498230e+00	 1.1510679e-01	  1.4026749e+00 	 1.7480373e-01


.. parsed-literal::

     132	 1.4970577e+00	 1.2371350e-01	 1.5512738e+00	 1.1498131e-01	  1.4024280e+00 	 2.1950817e-01
     133	 1.4977502e+00	 1.2363999e-01	 1.5519564e+00	 1.1491994e-01	  1.4021683e+00 	 1.7614245e-01


.. parsed-literal::

     134	 1.4986431e+00	 1.2354345e-01	 1.5528478e+00	 1.1490675e-01	  1.4015917e+00 	 1.7567205e-01


.. parsed-literal::

     135	 1.5001891e+00	 1.2335784e-01	 1.5544167e+00	 1.1501498e-01	  1.3995642e+00 	 2.0230126e-01


.. parsed-literal::

     136	 1.5010361e+00	 1.2331588e-01	 1.5552960e+00	 1.1507339e-01	  1.3989832e+00 	 3.2640791e-01


.. parsed-literal::

     137	 1.5025123e+00	 1.2314197e-01	 1.5568345e+00	 1.1523280e-01	  1.3964468e+00 	 2.0930982e-01


.. parsed-literal::

     138	 1.5036586e+00	 1.2307919e-01	 1.5580613e+00	 1.1534072e-01	  1.3946369e+00 	 2.0859718e-01
     139	 1.5046681e+00	 1.2297058e-01	 1.5591647e+00	 1.1526282e-01	  1.3929084e+00 	 1.9275308e-01


.. parsed-literal::

     140	 1.5056774e+00	 1.2301451e-01	 1.5602198e+00	 1.1519361e-01	  1.3925585e+00 	 1.9073892e-01
     141	 1.5067794e+00	 1.2301693e-01	 1.5613590e+00	 1.1507677e-01	  1.3912432e+00 	 1.9002366e-01


.. parsed-literal::

     142	 1.5078514e+00	 1.2300516e-01	 1.5624827e+00	 1.1493435e-01	  1.3889401e+00 	 2.0788479e-01
     143	 1.5088063e+00	 1.2296638e-01	 1.5634677e+00	 1.1494490e-01	  1.3857574e+00 	 1.8544292e-01


.. parsed-literal::

     144	 1.5096250e+00	 1.2291271e-01	 1.5642765e+00	 1.1497560e-01	  1.3845793e+00 	 2.0182085e-01


.. parsed-literal::

     145	 1.5105644e+00	 1.2281967e-01	 1.5652301e+00	 1.1506340e-01	  1.3819202e+00 	 2.1047115e-01


.. parsed-literal::

     146	 1.5114960e+00	 1.2280777e-01	 1.5661836e+00	 1.1508389e-01	  1.3800204e+00 	 2.1206808e-01


.. parsed-literal::

     147	 1.5128775e+00	 1.2268851e-01	 1.5676385e+00	 1.1518724e-01	  1.3727913e+00 	 2.1052933e-01


.. parsed-literal::

     148	 1.5137503e+00	 1.2267223e-01	 1.5685689e+00	 1.1514482e-01	  1.3712368e+00 	 2.0608902e-01
     149	 1.5146695e+00	 1.2259142e-01	 1.5694577e+00	 1.1511701e-01	  1.3702962e+00 	 1.7040586e-01


.. parsed-literal::

     150	 1.5156152e+00	 1.2243858e-01	 1.5703883e+00	 1.1505662e-01	  1.3687172e+00 	 1.8210459e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.24 s, total: 2min 8s
    Wall time: 32.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f357c61cb80>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.88 s, sys: 49 ms, total: 1.92 s
    Wall time: 656 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

