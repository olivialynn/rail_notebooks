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
    from rail.core.utils import find_rail_file
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
       1	-3.0478501e-01	 3.2320466e-01	-2.9068276e-01	 2.9708180e-01	[-2.1996444e-01]	 4.9229836e-01
       2	-2.3609585e-01	 2.9283388e-01	-2.0065130e-01	 2.7709491e-01	[-1.3706579e-01]	 1.1730695e-01


.. parsed-literal::

       3	-1.6103405e-01	 2.6779042e-01	-1.3196855e-01	 2.6026863e-01	[-9.2126152e-02]	 1.1859918e-01
       4	-8.8873968e-02	 2.5792054e-01	-6.4947316e-02	 2.4718221e-01	[-2.6684540e-02]	 1.1681199e-01


.. parsed-literal::

       5	-5.9823602e-02	 2.5155557e-01	-3.9325145e-02	 2.3922970e-01	[ 9.7688099e-03]	 1.1955309e-01
       6	-4.0726219e-02	 2.4746140e-01	-2.2514251e-02	 2.3579993e-01	[ 2.2817185e-02]	 1.1696649e-01


.. parsed-literal::

       7	-3.0106454e-02	 2.4517157e-01	-1.3375434e-02	 2.3453662e-01	[ 2.9949246e-02]	 1.1883664e-01
       8	-2.0212713e-02	 2.4306453e-01	-4.2576891e-03	 2.3306249e-01	[ 3.5160874e-02]	 1.1954522e-01


.. parsed-literal::

       9	-1.4078230e-02	 2.4178819e-01	 1.3516241e-03	 2.3241873e-01	[ 3.9273340e-02]	 1.1927581e-01
      10	-9.5492131e-03	 2.4084386e-01	 5.4521088e-03	 2.3206486e-01	[ 4.0081985e-02]	 1.1945724e-01


.. parsed-literal::

      11	-3.3970473e-03	 2.3931393e-01	 1.1975877e-02	 2.3151047e-01	[ 4.2661856e-02]	 1.2072897e-01
      12	 1.7254005e-02	 2.3489351e-01	 3.3944671e-02	 2.2943903e-01	[ 5.5717024e-02]	 1.1816478e-01


.. parsed-literal::

      13	 5.7112515e-02	 2.2895258e-01	 7.5928534e-02	 2.2494191e-01	[ 9.3494006e-02]	 1.2367439e-01
      14	 1.2741044e-01	 2.3142459e-01	 1.5726981e-01	 2.2668118e-01	[ 1.7338305e-01]	 1.1789441e-01


.. parsed-literal::

      15	 2.0562151e-01	 2.2357094e-01	 2.3716537e-01	 2.2202794e-01	[ 2.4309615e-01]	 1.2120366e-01
      16	 2.7541570e-01	 2.1329259e-01	 3.1137569e-01	 2.1039516e-01	[ 3.2066561e-01]	 1.2726188e-01


.. parsed-literal::

      17	 2.9880267e-01	 2.0902496e-01	 3.3426399e-01	 2.0597852e-01	[ 3.4046946e-01]	 1.2040973e-01
      18	 3.2745612e-01	 2.0648941e-01	 3.6294892e-01	 2.0722181e-01	[ 3.5886559e-01]	 1.1824512e-01


.. parsed-literal::

      19	 3.6471762e-01	 2.0059325e-01	 4.0158023e-01	 2.0392487e-01	[ 3.8791643e-01]	 1.1986017e-01
      20	 4.0888538e-01	 1.9566650e-01	 4.4689319e-01	 1.9683662e-01	[ 4.3398038e-01]	 1.1900139e-01


.. parsed-literal::

      21	 4.7638151e-01	 1.8945325e-01	 5.1603070e-01	 1.9006467e-01	[ 4.9964515e-01]	 1.1916089e-01
      22	 5.5199977e-01	 1.8473407e-01	 5.9027434e-01	 1.8536439e-01	[ 5.8636642e-01]	 1.1770558e-01


.. parsed-literal::

      23	 6.0196687e-01	 1.8558130e-01	 6.4047316e-01	 1.8595727e-01	[ 6.4213248e-01]	 1.2008333e-01
      24	 6.3705694e-01	 1.8376279e-01	 6.7735859e-01	 1.8247272e-01	[ 6.8461750e-01]	 1.1889887e-01


.. parsed-literal::

      25	 6.5643874e-01	 1.8263014e-01	 6.9549947e-01	 1.8067867e-01	[ 7.0244414e-01]	 1.2773824e-01
      26	 6.8762145e-01	 1.8098574e-01	 7.2823015e-01	 1.7912633e-01	[ 7.3628020e-01]	 1.1915827e-01


.. parsed-literal::

      27	 7.1421301e-01	 1.8064910e-01	 7.5477097e-01	 1.8098769e-01	[ 7.5997579e-01]	 2.3759127e-01
      28	 7.3933661e-01	 1.7911091e-01	 7.7887607e-01	 1.8154770e-01	[ 7.7772890e-01]	 1.1714864e-01


.. parsed-literal::

      29	 7.6651360e-01	 1.7850331e-01	 8.0633023e-01	 1.8167531e-01	[ 8.0169435e-01]	 1.1932158e-01
      30	 7.9822046e-01	 1.7699049e-01	 8.3900358e-01	 1.7827920e-01	[ 8.4153718e-01]	 1.2374115e-01


.. parsed-literal::

      31	 8.1725887e-01	 1.7973797e-01	 8.5903632e-01	 1.8159017e-01	[ 8.5935838e-01]	 2.4383640e-01
      32	 8.3891493e-01	 1.7851774e-01	 8.8178616e-01	 1.7899451e-01	[ 8.8215011e-01]	 1.1764979e-01


.. parsed-literal::

      33	 8.6230855e-01	 1.7619525e-01	 9.0573731e-01	 1.7646276e-01	[ 9.0619960e-01]	 1.1959982e-01
      34	 8.9391654e-01	 1.7735854e-01	 9.3893977e-01	 1.7405306e-01	[ 9.3793602e-01]	 1.1655188e-01


.. parsed-literal::

      35	 9.1632611e-01	 1.7407104e-01	 9.6097771e-01	 1.6955689e-01	[ 9.6841270e-01]	 1.1943436e-01
      36	 9.3822137e-01	 1.7467054e-01	 9.8275206e-01	 1.7048139e-01	[ 9.8068140e-01]	 1.1729527e-01


.. parsed-literal::

      37	 9.4447367e-01	 1.7466886e-01	 9.8960439e-01	 1.6860923e-01	[ 9.9256950e-01]	 1.1937809e-01
      38	 9.6208615e-01	 1.7187998e-01	 1.0069724e+00	 1.6658380e-01	[ 1.0081344e+00]	 1.1797142e-01


.. parsed-literal::

      39	 9.7325311e-01	 1.7008372e-01	 1.0184658e+00	 1.6544809e-01	[ 1.0164841e+00]	 1.2034106e-01
      40	 9.8574372e-01	 1.6781319e-01	 1.0314689e+00	 1.6342342e-01	[ 1.0293851e+00]	 1.2555122e-01


.. parsed-literal::

      41	 1.0045188e+00	 1.6491898e-01	 1.0507517e+00	 1.6114287e-01	[ 1.0483251e+00]	 1.2001348e-01
      42	 1.0165517e+00	 1.6532124e-01	 1.0645352e+00	 1.5918827e-01	[ 1.0659198e+00]	 1.1731410e-01


.. parsed-literal::

      43	 1.0352124e+00	 1.6079129e-01	 1.0826360e+00	 1.5717996e-01	[ 1.0876493e+00]	 1.2010312e-01
      44	 1.0429095e+00	 1.5957200e-01	 1.0897211e+00	 1.5675784e-01	[ 1.0925732e+00]	 1.1757898e-01


.. parsed-literal::

      45	 1.0528884e+00	 1.5830846e-01	 1.0997373e+00	 1.5633277e-01	[ 1.0980884e+00]	 1.1992073e-01
      46	 1.0695827e+00	 1.5810125e-01	 1.1176513e+00	 1.5739212e-01	[ 1.1064950e+00]	 1.1730385e-01


.. parsed-literal::

      47	 1.0719137e+00	 1.5878105e-01	 1.1201751e+00	 1.5662600e-01	[ 1.1080075e+00]	 1.1960530e-01
      48	 1.0847460e+00	 1.5716222e-01	 1.1327833e+00	 1.5561827e-01	[ 1.1200530e+00]	 1.2454534e-01


.. parsed-literal::

      49	 1.0902394e+00	 1.5670926e-01	 1.1385246e+00	 1.5549530e-01	[ 1.1231051e+00]	 1.2030053e-01
      50	 1.1008614e+00	 1.5586596e-01	 1.1496188e+00	 1.5526985e-01	[ 1.1288233e+00]	 1.1904836e-01


.. parsed-literal::

      51	 1.1099302e+00	 1.5481958e-01	 1.1590111e+00	 1.5438735e-01	[ 1.1331543e+00]	 1.2056375e-01
      52	 1.1214775e+00	 1.5266853e-01	 1.1711556e+00	 1.5252728e-01	[ 1.1456586e+00]	 1.1919665e-01


.. parsed-literal::

      53	 1.1325211e+00	 1.5167583e-01	 1.1821958e+00	 1.5111988e-01	[ 1.1554916e+00]	 1.1963367e-01
      54	 1.1401746e+00	 1.5081102e-01	 1.1898693e+00	 1.5031822e-01	[ 1.1639139e+00]	 1.1823964e-01


.. parsed-literal::

      55	 1.1504264e+00	 1.4881068e-01	 1.2007072e+00	 1.4897252e-01	[ 1.1746042e+00]	 1.2038827e-01
      56	 1.1557322e+00	 1.4844105e-01	 1.2062199e+00	 1.4841426e-01	[ 1.1767765e+00]	 1.2537384e-01


.. parsed-literal::

      57	 1.1626357e+00	 1.4757345e-01	 1.2129164e+00	 1.4773706e-01	[ 1.1851732e+00]	 1.1965775e-01
      58	 1.1702926e+00	 1.4642591e-01	 1.2206026e+00	 1.4663248e-01	[ 1.1922754e+00]	 1.1765528e-01


.. parsed-literal::

      59	 1.1769215e+00	 1.4537350e-01	 1.2274177e+00	 1.4507165e-01	[ 1.1964159e+00]	 1.1956906e-01


.. parsed-literal::

      60	 1.1828878e+00	 1.4423088e-01	 1.2336220e+00	 1.4404409e-01	[ 1.1983808e+00]	 2.3649454e-01
      61	 1.1912203e+00	 1.4335169e-01	 1.2421436e+00	 1.4261801e-01	[ 1.2040273e+00]	 1.1739826e-01


.. parsed-literal::

      62	 1.1988959e+00	 1.4271573e-01	 1.2500383e+00	 1.4169518e-01	[ 1.2095538e+00]	 1.2021804e-01
      63	 1.2078317e+00	 1.4215275e-01	 1.2592163e+00	 1.4103158e-01	[ 1.2153440e+00]	 1.1825991e-01


.. parsed-literal::

      64	 1.2172652e+00	 1.4144966e-01	 1.2689148e+00	 1.4023862e-01	[ 1.2214964e+00]	 1.2653232e-01
      65	 1.2236407e+00	 1.4132406e-01	 1.2756209e+00	 1.3956536e-01	[ 1.2236383e+00]	 1.1846542e-01


.. parsed-literal::

      66	 1.2289374e+00	 1.4085512e-01	 1.2806668e+00	 1.3920408e-01	[ 1.2338035e+00]	 1.2009478e-01
      67	 1.2344498e+00	 1.4014412e-01	 1.2862813e+00	 1.3836971e-01	[ 1.2430468e+00]	 1.1851120e-01


.. parsed-literal::

      68	 1.2405137e+00	 1.3952860e-01	 1.2926157e+00	 1.3687656e-01	[ 1.2531918e+00]	 1.2002683e-01
      69	 1.2437850e+00	 1.3870162e-01	 1.2964927e+00	 1.3598226e-01	[ 1.2569261e+00]	 1.1757469e-01


.. parsed-literal::

      70	 1.2518447e+00	 1.3850641e-01	 1.3044018e+00	 1.3492412e-01	[ 1.2651935e+00]	 1.1887288e-01
      71	 1.2559300e+00	 1.3847039e-01	 1.3085121e+00	 1.3444220e-01	[ 1.2672782e+00]	 1.1832929e-01


.. parsed-literal::

      72	 1.2608389e+00	 1.3833041e-01	 1.3135983e+00	 1.3396996e-01	[ 1.2679385e+00]	 1.2763476e-01
      73	 1.2685520e+00	 1.3785178e-01	 1.3216547e+00	 1.3340163e-01	  1.2675549e+00 	 1.1838889e-01


.. parsed-literal::

      74	 1.2790170e+00	 1.3703974e-01	 1.3326981e+00	 1.3162117e-01	[ 1.2691155e+00]	 1.1915255e-01
      75	 1.2841423e+00	 1.3679766e-01	 1.3382667e+00	 1.3059450e-01	  1.2648717e+00 	 1.7868876e-01


.. parsed-literal::

      76	 1.2912115e+00	 1.3652639e-01	 1.3449892e+00	 1.3098160e-01	[ 1.2760526e+00]	 1.1996555e-01
      77	 1.2972095e+00	 1.3604393e-01	 1.3509658e+00	 1.3006701e-01	[ 1.2858390e+00]	 1.1794472e-01


.. parsed-literal::

      78	 1.3048935e+00	 1.3558804e-01	 1.3589600e+00	 1.2890210e-01	[ 1.2932414e+00]	 1.1870742e-01
      79	 1.3078973e+00	 1.3567383e-01	 1.3621781e+00	 1.2767328e-01	[ 1.3043878e+00]	 1.1837554e-01


.. parsed-literal::

      80	 1.3169496e+00	 1.3523956e-01	 1.3710976e+00	 1.2705963e-01	[ 1.3062074e+00]	 1.2766385e-01
      81	 1.3201260e+00	 1.3505966e-01	 1.3742897e+00	 1.2717611e-01	  1.3055294e+00 	 1.1851478e-01


.. parsed-literal::

      82	 1.3262699e+00	 1.3467839e-01	 1.3806762e+00	 1.2736653e-01	  1.3043226e+00 	 1.1913943e-01
      83	 1.3287170e+00	 1.3493216e-01	 1.3834951e+00	 1.2689777e-01	  1.2970464e+00 	 1.1870956e-01


.. parsed-literal::

      84	 1.3349354e+00	 1.3452452e-01	 1.3895587e+00	 1.2706185e-01	  1.3059063e+00 	 1.1974430e-01
      85	 1.3381466e+00	 1.3442410e-01	 1.3927350e+00	 1.2693740e-01	[ 1.3112355e+00]	 1.1848712e-01


.. parsed-literal::

      86	 1.3418713e+00	 1.3428264e-01	 1.3964601e+00	 1.2667193e-01	[ 1.3156601e+00]	 1.2063026e-01
      87	 1.3483346e+00	 1.3395613e-01	 1.4030159e+00	 1.2612481e-01	[ 1.3206909e+00]	 1.1816597e-01


.. parsed-literal::

      88	 1.3523558e+00	 1.3355895e-01	 1.4072637e+00	 1.2563141e-01	[ 1.3209896e+00]	 2.4426961e-01
      89	 1.3584381e+00	 1.3337190e-01	 1.4135120e+00	 1.2516136e-01	[ 1.3242026e+00]	 1.1733818e-01


.. parsed-literal::

      90	 1.3632264e+00	 1.3310532e-01	 1.4183231e+00	 1.2500020e-01	[ 1.3272470e+00]	 1.1985278e-01
      91	 1.3679511e+00	 1.3278416e-01	 1.4233002e+00	 1.2494835e-01	[ 1.3303898e+00]	 1.1769629e-01


.. parsed-literal::

      92	 1.3721306e+00	 1.3250747e-01	 1.4276226e+00	 1.2502415e-01	  1.3293114e+00 	 1.2299633e-01
      93	 1.3769239e+00	 1.3227832e-01	 1.4324746e+00	 1.2517235e-01	[ 1.3337361e+00]	 1.1854124e-01


.. parsed-literal::

      94	 1.3834554e+00	 1.3212839e-01	 1.4393146e+00	 1.2512335e-01	[ 1.3382541e+00]	 1.2009573e-01
      95	 1.3871887e+00	 1.3218794e-01	 1.4430235e+00	 1.2474273e-01	[ 1.3418053e+00]	 1.1831450e-01


.. parsed-literal::

      96	 1.3902161e+00	 1.3222183e-01	 1.4459751e+00	 1.2452775e-01	[ 1.3446138e+00]	 1.2624002e-01
      97	 1.3957998e+00	 1.3212347e-01	 1.4516225e+00	 1.2362129e-01	  1.3440345e+00 	 1.2168145e-01


.. parsed-literal::

      98	 1.3993145e+00	 1.3195736e-01	 1.4552436e+00	 1.2390383e-01	  1.3418777e+00 	 1.2153292e-01
      99	 1.4027955e+00	 1.3168805e-01	 1.4586780e+00	 1.2374142e-01	  1.3431725e+00 	 1.1774707e-01


.. parsed-literal::

     100	 1.4090695e+00	 1.3108439e-01	 1.4649709e+00	 1.2348555e-01	  1.3444091e+00 	 1.1988258e-01
     101	 1.4104860e+00	 1.3043224e-01	 1.4665531e+00	 1.2389368e-01	  1.3398503e+00 	 1.1755157e-01


.. parsed-literal::

     102	 1.4141237e+00	 1.3037847e-01	 1.4700303e+00	 1.2351249e-01	[ 1.3462624e+00]	 1.2002087e-01
     103	 1.4172080e+00	 1.3030808e-01	 1.4730880e+00	 1.2317445e-01	[ 1.3498201e+00]	 1.1858153e-01


.. parsed-literal::

     104	 1.4201466e+00	 1.3006656e-01	 1.4760483e+00	 1.2287212e-01	[ 1.3516871e+00]	 1.2800956e-01
     105	 1.4246030e+00	 1.2967604e-01	 1.4804920e+00	 1.2244803e-01	[ 1.3543461e+00]	 1.1748886e-01


.. parsed-literal::

     106	 1.4271507e+00	 1.2938846e-01	 1.4831149e+00	 1.2260052e-01	  1.3495976e+00 	 2.3775649e-01
     107	 1.4299181e+00	 1.2923692e-01	 1.4859047e+00	 1.2262948e-01	  1.3502293e+00 	 1.1793590e-01


.. parsed-literal::

     108	 1.4324393e+00	 1.2902476e-01	 1.4885008e+00	 1.2283251e-01	  1.3501342e+00 	 1.2032580e-01
     109	 1.4359182e+00	 1.2871442e-01	 1.4921810e+00	 1.2306484e-01	  1.3507759e+00 	 1.1839485e-01


.. parsed-literal::

     110	 1.4393486e+00	 1.2842822e-01	 1.4957765e+00	 1.2334356e-01	  1.3489681e+00 	 1.2002707e-01
     111	 1.4423757e+00	 1.2830722e-01	 1.4987934e+00	 1.2313666e-01	  1.3524348e+00 	 1.1783028e-01


.. parsed-literal::

     112	 1.4453007e+00	 1.2823538e-01	 1.5017305e+00	 1.2252944e-01	[ 1.3556601e+00]	 1.2776542e-01
     113	 1.4472849e+00	 1.2806929e-01	 1.5037722e+00	 1.2252544e-01	[ 1.3560039e+00]	 1.1775517e-01


.. parsed-literal::

     114	 1.4496763e+00	 1.2786736e-01	 1.5062307e+00	 1.2240086e-01	  1.3552598e+00 	 1.1984134e-01
     115	 1.4536832e+00	 1.2742096e-01	 1.5104004e+00	 1.2213662e-01	  1.3528053e+00 	 1.1808038e-01


.. parsed-literal::

     116	 1.4557875e+00	 1.2708594e-01	 1.5125320e+00	 1.2208145e-01	  1.3534877e+00 	 2.3909163e-01
     117	 1.4577472e+00	 1.2688561e-01	 1.5144298e+00	 1.2194376e-01	[ 1.3568456e+00]	 1.1903214e-01


.. parsed-literal::

     118	 1.4615386e+00	 1.2635360e-01	 1.5182383e+00	 1.2173092e-01	[ 1.3602233e+00]	 1.2178707e-01
     119	 1.4629695e+00	 1.2607766e-01	 1.5196220e+00	 1.2148613e-01	[ 1.3659281e+00]	 1.2583232e-01


.. parsed-literal::

     120	 1.4650399e+00	 1.2596965e-01	 1.5216583e+00	 1.2148847e-01	  1.3647974e+00 	 1.2107539e-01
     121	 1.4670101e+00	 1.2581579e-01	 1.5236268e+00	 1.2144515e-01	  1.3636650e+00 	 1.1896706e-01


.. parsed-literal::

     122	 1.4688046e+00	 1.2562858e-01	 1.5254196e+00	 1.2133520e-01	  1.3627260e+00 	 1.2073708e-01
     123	 1.4707937e+00	 1.2535573e-01	 1.5274997e+00	 1.2098867e-01	  1.3610346e+00 	 1.1731124e-01


.. parsed-literal::

     124	 1.4736655e+00	 1.2511230e-01	 1.5302903e+00	 1.2095253e-01	  1.3588348e+00 	 1.2124443e-01
     125	 1.4752913e+00	 1.2495631e-01	 1.5319037e+00	 1.2093093e-01	  1.3581259e+00 	 1.1885977e-01


.. parsed-literal::

     126	 1.4779677e+00	 1.2467883e-01	 1.5346252e+00	 1.2088315e-01	  1.3555026e+00 	 1.2150431e-01
     127	 1.4809274e+00	 1.2430824e-01	 1.5376567e+00	 1.2098652e-01	  1.3499251e+00 	 1.2718225e-01


.. parsed-literal::

     128	 1.4827651e+00	 1.2417997e-01	 1.5396019e+00	 1.2099332e-01	  1.3459085e+00 	 2.3698616e-01
     129	 1.4846869e+00	 1.2405557e-01	 1.5415609e+00	 1.2111576e-01	  1.3442138e+00 	 1.1880994e-01


.. parsed-literal::

     130	 1.4862265e+00	 1.2400446e-01	 1.5431666e+00	 1.2117463e-01	  1.3438623e+00 	 1.2173390e-01
     131	 1.4881900e+00	 1.2398877e-01	 1.5452519e+00	 1.2119411e-01	  1.3434781e+00 	 1.1953402e-01


.. parsed-literal::

     132	 1.4904557e+00	 1.2397319e-01	 1.5476414e+00	 1.2134785e-01	  1.3452644e+00 	 1.2032700e-01
     133	 1.4924435e+00	 1.2390949e-01	 1.5496655e+00	 1.2130628e-01	  1.3439339e+00 	 1.1930323e-01


.. parsed-literal::

     134	 1.4943990e+00	 1.2378081e-01	 1.5515848e+00	 1.2124770e-01	  1.3439066e+00 	 1.1980295e-01
     135	 1.4961211e+00	 1.2365547e-01	 1.5532575e+00	 1.2120603e-01	  1.3418998e+00 	 1.2652755e-01


.. parsed-literal::

     136	 1.4985244e+00	 1.2360887e-01	 1.5556621e+00	 1.2103744e-01	  1.3385238e+00 	 1.2069798e-01
     137	 1.5005829e+00	 1.2349989e-01	 1.5577342e+00	 1.2104617e-01	  1.3341065e+00 	 1.1882019e-01


.. parsed-literal::

     138	 1.5024017e+00	 1.2354111e-01	 1.5596174e+00	 1.2096591e-01	  1.3327873e+00 	 1.1960602e-01
     139	 1.5048164e+00	 1.2360756e-01	 1.5621944e+00	 1.2065154e-01	  1.3293705e+00 	 1.1731839e-01


.. parsed-literal::

     140	 1.5067343e+00	 1.2363592e-01	 1.5640496e+00	 1.2041523e-01	  1.3275890e+00 	 1.1950183e-01
     141	 1.5080414e+00	 1.2357734e-01	 1.5652968e+00	 1.2032277e-01	  1.3281175e+00 	 1.1814332e-01


.. parsed-literal::

     142	 1.5108399e+00	 1.2346836e-01	 1.5680780e+00	 1.2023943e-01	  1.3276290e+00 	 1.1943841e-01
     143	 1.5111708e+00	 1.2346533e-01	 1.5685664e+00	 1.2003886e-01	  1.3225485e+00 	 1.2438703e-01


.. parsed-literal::

     144	 1.5132113e+00	 1.2336548e-01	 1.5705291e+00	 1.2016676e-01	  1.3263407e+00 	 1.2256122e-01
     145	 1.5141932e+00	 1.2331810e-01	 1.5715485e+00	 1.2022800e-01	  1.3266237e+00 	 1.1790538e-01


.. parsed-literal::

     146	 1.5157388e+00	 1.2318274e-01	 1.5732103e+00	 1.2034278e-01	  1.3260463e+00 	 1.1979604e-01
     147	 1.5167490e+00	 1.2304201e-01	 1.5744054e+00	 1.2036534e-01	  1.3235223e+00 	 1.1781454e-01


.. parsed-literal::

     148	 1.5181974e+00	 1.2298173e-01	 1.5758583e+00	 1.2033464e-01	  1.3239518e+00 	 1.2026620e-01
     149	 1.5189869e+00	 1.2292798e-01	 1.5766432e+00	 1.2025564e-01	  1.3231269e+00 	 1.1800790e-01


.. parsed-literal::

     150	 1.5199313e+00	 1.2285248e-01	 1.5776267e+00	 1.2010932e-01	  1.3217953e+00 	 1.2072492e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 35.7 s, sys: 41.8 s, total: 1min 17s
    Wall time: 19.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f9dc7f71720>



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
    Process 0 running estimator on chunk 0 - 10000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10000 - 20000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20000 - 20449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 808 ms, sys: 1.03 s, total: 1.84 s
    Wall time: 597 ms


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




.. image:: ../../../docs/rendered/estimation_examples/GPz_estimation_example_files/../../../docs/rendered/estimation_examples/GPz_estimation_example_16_1.png


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




.. image:: ../../../docs/rendered/estimation_examples/GPz_estimation_example_files/../../../docs/rendered/estimation_examples/GPz_estimation_example_19_1.png

