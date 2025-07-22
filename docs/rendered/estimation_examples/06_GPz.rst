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
       1	-3.5251545e-01	 3.2324200e-01	-3.4276351e-01	 3.0956318e-01	[-3.1799871e-01]	 4.6008801e-01


.. parsed-literal::

       2	-2.7852954e-01	 3.1136806e-01	-2.5375987e-01	 2.9963781e-01	[-2.1975507e-01]	 2.3276973e-01


.. parsed-literal::

       3	-2.3614381e-01	 2.9231417e-01	-1.9535957e-01	 2.8298664e-01	[-1.5880918e-01]	 2.9775667e-01
       4	-1.9836334e-01	 2.6741713e-01	-1.5740044e-01	 2.5727811e-01	[-1.0212256e-01]	 1.8251395e-01


.. parsed-literal::

       5	-1.1365587e-01	 2.5894421e-01	-7.8343333e-02	 2.4910878e-01	[-3.3386728e-02]	 2.1836400e-01
       6	-7.8404260e-02	 2.5333929e-01	-4.6675985e-02	 2.4405511e-01	[-1.3566278e-02]	 1.8867230e-01


.. parsed-literal::

       7	-6.0882881e-02	 2.5073467e-01	-3.5868405e-02	 2.4141823e-01	[-4.1589915e-04]	 2.0678782e-01


.. parsed-literal::

       8	-4.7758279e-02	 2.4858692e-01	-2.7000304e-02	 2.3949677e-01	[ 8.2398757e-03]	 2.0745707e-01


.. parsed-literal::

       9	-3.4295114e-02	 2.4616037e-01	-1.6756872e-02	 2.3701355e-01	[ 1.9544375e-02]	 2.0351076e-01


.. parsed-literal::

      10	-2.3232996e-02	 2.4411959e-01	-7.9690407e-03	 2.3440739e-01	[ 2.9175566e-02]	 2.0561910e-01
      11	-1.8176480e-02	 2.4317391e-01	-4.1914161e-03	 2.3442837e-01	[ 3.3268388e-02]	 1.8284774e-01


.. parsed-literal::

      12	-1.5128282e-02	 2.4271657e-01	-1.2587060e-03	 2.3415622e-01	  3.2598310e-02 	 1.9906592e-01


.. parsed-literal::

      13	-1.2651357e-02	 2.4218515e-01	 1.1821156e-03	 2.3367596e-01	[ 3.4538850e-02]	 2.0594335e-01
      14	-8.4331743e-03	 2.4131957e-01	 5.7501565e-03	 2.3334775e-01	[ 3.6767560e-02]	 2.0076656e-01


.. parsed-literal::

      15	 1.1013959e-01	 2.2666949e-01	 1.3212750e-01	 2.1983393e-01	[ 1.4700258e-01]	 2.9699588e-01


.. parsed-literal::

      16	 1.3426426e-01	 2.2323005e-01	 1.5804976e-01	 2.1694339e-01	[ 1.6505680e-01]	 2.1516347e-01


.. parsed-literal::

      17	 2.3842557e-01	 2.1815340e-01	 2.6494479e-01	 2.1252624e-01	[ 2.7281103e-01]	 2.0684099e-01
      18	 2.6506986e-01	 2.1638920e-01	 3.0027360e-01	 2.0905803e-01	[ 3.2327509e-01]	 1.9882774e-01


.. parsed-literal::

      19	 3.3465619e-01	 2.1188034e-01	 3.6823125e-01	 2.0667938e-01	[ 3.7867887e-01]	 2.1126246e-01
      20	 3.7239388e-01	 2.0794417e-01	 4.0670974e-01	 2.0458080e-01	[ 4.1197026e-01]	 1.7776895e-01


.. parsed-literal::

      21	 4.0756537e-01	 2.0616020e-01	 4.4120660e-01	 2.0246548e-01	[ 4.4337052e-01]	 2.0729852e-01
      22	 4.5747352e-01	 2.0487350e-01	 4.9149393e-01	 2.0023506e-01	[ 4.8530293e-01]	 1.7941236e-01


.. parsed-literal::

      23	 5.3019235e-01	 2.0025251e-01	 5.6613002e-01	 1.9533536e-01	[ 5.5380673e-01]	 2.1564579e-01
      24	 6.0355233e-01	 1.9710550e-01	 6.4240904e-01	 1.9242336e-01	[ 6.2609281e-01]	 1.8093085e-01


.. parsed-literal::

      25	 6.4005914e-01	 1.9134227e-01	 6.7853267e-01	 1.8583319e-01	[ 6.6024515e-01]	 1.9904327e-01


.. parsed-literal::

      26	 6.6596975e-01	 1.8976352e-01	 7.0423791e-01	 1.8550768e-01	[ 6.8508613e-01]	 2.1849275e-01
      27	 6.9364981e-01	 1.9746026e-01	 7.3158198e-01	 1.9660338e-01	[ 7.1226849e-01]	 1.8100762e-01


.. parsed-literal::

      28	 7.3352468e-01	 1.9620904e-01	 7.7095398e-01	 1.9381249e-01	[ 7.5510708e-01]	 1.9484425e-01


.. parsed-literal::

      29	 7.6429590e-01	 1.9535702e-01	 8.0326794e-01	 1.9226453e-01	[ 7.8486800e-01]	 2.1821952e-01


.. parsed-literal::

      30	 8.0991493e-01	 1.9274665e-01	 8.4999929e-01	 1.8967367e-01	[ 8.2372811e-01]	 2.1610689e-01


.. parsed-literal::

      31	 8.3630567e-01	 1.9514450e-01	 8.7724775e-01	 1.9102319e-01	[ 8.5472899e-01]	 2.1409345e-01


.. parsed-literal::

      32	 8.6232686e-01	 1.9354630e-01	 9.0408152e-01	 1.8880438e-01	[ 8.8785812e-01]	 2.0562530e-01


.. parsed-literal::

      33	 8.8251789e-01	 1.9595086e-01	 9.2414712e-01	 1.9139844e-01	[ 9.1542854e-01]	 2.0739150e-01
      34	 8.9607033e-01	 1.9048246e-01	 9.3938413e-01	 1.8546256e-01	[ 9.4318931e-01]	 1.9005990e-01


.. parsed-literal::

      35	 9.1879766e-01	 1.8851386e-01	 9.6132454e-01	 1.8429522e-01	[ 9.6080829e-01]	 1.8352437e-01


.. parsed-literal::

      36	 9.3797100e-01	 1.8503149e-01	 9.8061512e-01	 1.8150650e-01	[ 9.7972304e-01]	 2.0177698e-01
      37	 9.5978739e-01	 1.8067321e-01	 1.0029274e+00	 1.7741750e-01	[ 1.0025873e+00]	 1.9294930e-01


.. parsed-literal::

      38	 9.8847605e-01	 1.7348641e-01	 1.0337490e+00	 1.6982688e-01	[ 1.0234865e+00]	 2.1044970e-01


.. parsed-literal::

      39	 1.0136025e+00	 1.7051919e-01	 1.0592975e+00	 1.6716862e-01	[ 1.0355338e+00]	 2.1842670e-01


.. parsed-literal::

      40	 1.0280068e+00	 1.6899753e-01	 1.0736010e+00	 1.6615729e-01	[ 1.0477073e+00]	 2.1035671e-01


.. parsed-literal::

      41	 1.0453377e+00	 1.6859372e-01	 1.0916731e+00	 1.6629112e-01	[ 1.0537429e+00]	 2.1643519e-01


.. parsed-literal::

      42	 1.0561594e+00	 1.6576871e-01	 1.1027214e+00	 1.6311838e-01	[ 1.0608426e+00]	 2.0379615e-01


.. parsed-literal::

      43	 1.0671963e+00	 1.6422926e-01	 1.1132728e+00	 1.6198173e-01	[ 1.0677408e+00]	 2.1132803e-01


.. parsed-literal::

      44	 1.0828256e+00	 1.6146832e-01	 1.1289766e+00	 1.5964155e-01	[ 1.0754361e+00]	 2.0249748e-01
      45	 1.0994439e+00	 1.5798943e-01	 1.1457358e+00	 1.5648061e-01	[ 1.0940931e+00]	 1.9957948e-01


.. parsed-literal::

      46	 1.1152839e+00	 1.5469507e-01	 1.1626129e+00	 1.5256581e-01	[ 1.1048391e+00]	 1.7577720e-01


.. parsed-literal::

      47	 1.1318888e+00	 1.5354647e-01	 1.1793102e+00	 1.5162124e-01	[ 1.1204775e+00]	 2.0932794e-01


.. parsed-literal::

      48	 1.1450843e+00	 1.5257543e-01	 1.1926418e+00	 1.5003550e-01	[ 1.1271386e+00]	 2.1052980e-01
      49	 1.1603858e+00	 1.5062466e-01	 1.2083025e+00	 1.4798211e-01	[ 1.1274012e+00]	 1.7876887e-01


.. parsed-literal::

      50	 1.1778302e+00	 1.4741901e-01	 1.2259589e+00	 1.4504236e-01	[ 1.1274305e+00]	 2.1338439e-01


.. parsed-literal::

      51	 1.1899019e+00	 1.4493354e-01	 1.2383250e+00	 1.4338461e-01	  1.1225020e+00 	 2.0584989e-01
      52	 1.2032189e+00	 1.4318839e-01	 1.2514963e+00	 1.4261087e-01	  1.1156282e+00 	 1.6700602e-01


.. parsed-literal::

      53	 1.2154174e+00	 1.4311979e-01	 1.2639317e+00	 1.4159316e-01	[ 1.1294490e+00]	 2.0749283e-01
      54	 1.2273906e+00	 1.4171582e-01	 1.2765073e+00	 1.3995793e-01	  1.1287912e+00 	 2.0074797e-01


.. parsed-literal::

      55	 1.2361213e+00	 1.4180194e-01	 1.2851821e+00	 1.3908638e-01	[ 1.1430077e+00]	 2.0753956e-01


.. parsed-literal::

      56	 1.2482394e+00	 1.4117185e-01	 1.2975476e+00	 1.3755869e-01	[ 1.1521372e+00]	 2.0852327e-01
      57	 1.2540143e+00	 1.4119092e-01	 1.3035818e+00	 1.3656855e-01	  1.1439507e+00 	 2.0123577e-01


.. parsed-literal::

      58	 1.2624458e+00	 1.4031282e-01	 1.3118963e+00	 1.3626117e-01	  1.1507874e+00 	 2.0718837e-01


.. parsed-literal::

      59	 1.2679194e+00	 1.3965493e-01	 1.3174408e+00	 1.3603660e-01	  1.1507442e+00 	 2.0706224e-01


.. parsed-literal::

      60	 1.2756948e+00	 1.3916140e-01	 1.3254787e+00	 1.3592430e-01	  1.1503753e+00 	 2.0525217e-01
      61	 1.2867752e+00	 1.3873500e-01	 1.3369274e+00	 1.3542847e-01	  1.1512691e+00 	 1.7329383e-01


.. parsed-literal::

      62	 1.2945204e+00	 1.3963650e-01	 1.3452010e+00	 1.3598269e-01	  1.1396889e+00 	 1.9417048e-01


.. parsed-literal::

      63	 1.3056789e+00	 1.3844252e-01	 1.3562413e+00	 1.3482209e-01	[ 1.1562741e+00]	 2.1858788e-01


.. parsed-literal::

      64	 1.3110479e+00	 1.3772560e-01	 1.3615750e+00	 1.3419091e-01	[ 1.1641257e+00]	 2.0969892e-01


.. parsed-literal::

      65	 1.3192022e+00	 1.3705900e-01	 1.3697480e+00	 1.3417143e-01	[ 1.1713869e+00]	 2.1396136e-01


.. parsed-literal::

      66	 1.3253404e+00	 1.3574518e-01	 1.3762967e+00	 1.3353663e-01	  1.1542056e+00 	 2.1205974e-01


.. parsed-literal::

      67	 1.3323379e+00	 1.3595971e-01	 1.3832258e+00	 1.3405492e-01	  1.1619247e+00 	 2.0190382e-01
      68	 1.3374711e+00	 1.3591446e-01	 1.3883510e+00	 1.3449826e-01	  1.1591240e+00 	 1.9317913e-01


.. parsed-literal::

      69	 1.3441058e+00	 1.3576762e-01	 1.3951753e+00	 1.3506613e-01	  1.1460032e+00 	 2.1157169e-01


.. parsed-literal::

      70	 1.3503075e+00	 1.3459886e-01	 1.4014427e+00	 1.3517292e-01	  1.1263801e+00 	 2.0880270e-01
      71	 1.3577965e+00	 1.3468259e-01	 1.4089419e+00	 1.3566942e-01	  1.1238264e+00 	 1.8682265e-01


.. parsed-literal::

      72	 1.3622604e+00	 1.3437714e-01	 1.4133551e+00	 1.3502802e-01	  1.1319536e+00 	 1.7661953e-01
      73	 1.3679282e+00	 1.3383489e-01	 1.4193009e+00	 1.3440942e-01	  1.1258824e+00 	 1.9867873e-01


.. parsed-literal::

      74	 1.3726245e+00	 1.3368580e-01	 1.4241012e+00	 1.3410724e-01	  1.1183304e+00 	 2.0198536e-01


.. parsed-literal::

      75	 1.3777849e+00	 1.3293370e-01	 1.4291412e+00	 1.3372260e-01	  1.1116120e+00 	 2.0238757e-01
      76	 1.3819754e+00	 1.3263022e-01	 1.4333589e+00	 1.3390250e-01	  1.1016939e+00 	 2.0185757e-01


.. parsed-literal::

      77	 1.3862806e+00	 1.3229086e-01	 1.4377129e+00	 1.3376116e-01	  1.0946628e+00 	 1.7793918e-01
      78	 1.3918030e+00	 1.3204407e-01	 1.4434693e+00	 1.3323657e-01	  1.0819905e+00 	 1.8454576e-01


.. parsed-literal::

      79	 1.3962593e+00	 1.3159181e-01	 1.4480104e+00	 1.3237138e-01	  1.0818100e+00 	 2.0454454e-01
      80	 1.3995919e+00	 1.3169335e-01	 1.4512849e+00	 1.3198916e-01	  1.0946063e+00 	 1.7792821e-01


.. parsed-literal::

      81	 1.4048516e+00	 1.3179570e-01	 1.4566778e+00	 1.3160001e-01	  1.1048706e+00 	 2.1291327e-01


.. parsed-literal::

      82	 1.4091704e+00	 1.3154560e-01	 1.4612407e+00	 1.3072874e-01	  1.1207036e+00 	 2.0606351e-01


.. parsed-literal::

      83	 1.4131539e+00	 1.3122856e-01	 1.4651452e+00	 1.3085076e-01	  1.1247193e+00 	 2.0652723e-01
      84	 1.4163625e+00	 1.3083080e-01	 1.4683420e+00	 1.3076027e-01	  1.1313038e+00 	 1.8422198e-01


.. parsed-literal::

      85	 1.4198429e+00	 1.3053346e-01	 1.4718497e+00	 1.3046336e-01	  1.1439849e+00 	 2.0928693e-01


.. parsed-literal::

      86	 1.4259389e+00	 1.3008428e-01	 1.4780445e+00	 1.2923114e-01	[ 1.1784039e+00]	 2.0151424e-01


.. parsed-literal::

      87	 1.4301332e+00	 1.2978410e-01	 1.4824939e+00	 1.2852024e-01	[ 1.2000306e+00]	 2.1029925e-01
      88	 1.4332297e+00	 1.2983248e-01	 1.4854541e+00	 1.2860195e-01	[ 1.2032707e+00]	 1.8073082e-01


.. parsed-literal::

      89	 1.4365488e+00	 1.2969216e-01	 1.4888237e+00	 1.2836436e-01	[ 1.2087384e+00]	 2.1032023e-01
      90	 1.4398104e+00	 1.2956610e-01	 1.4921402e+00	 1.2839889e-01	[ 1.2192078e+00]	 2.0058036e-01


.. parsed-literal::

      91	 1.4434652e+00	 1.2926086e-01	 1.4958423e+00	 1.2836086e-01	[ 1.2318111e+00]	 2.1281505e-01
      92	 1.4465098e+00	 1.2928010e-01	 1.4987958e+00	 1.2838475e-01	[ 1.2466173e+00]	 1.9855785e-01


.. parsed-literal::

      93	 1.4494177e+00	 1.2885475e-01	 1.5016782e+00	 1.2798707e-01	[ 1.2589734e+00]	 1.8440437e-01


.. parsed-literal::

      94	 1.4523241e+00	 1.2879692e-01	 1.5046068e+00	 1.2774355e-01	[ 1.2658944e+00]	 2.1148968e-01


.. parsed-literal::

      95	 1.4570057e+00	 1.2846645e-01	 1.5095292e+00	 1.2696753e-01	[ 1.2802067e+00]	 2.0944738e-01


.. parsed-literal::

      96	 1.4581219e+00	 1.2815613e-01	 1.5108955e+00	 1.2661548e-01	[ 1.2892726e+00]	 2.0798063e-01
      97	 1.4612617e+00	 1.2808022e-01	 1.5138615e+00	 1.2669229e-01	  1.2888876e+00 	 1.9985414e-01


.. parsed-literal::

      98	 1.4629225e+00	 1.2795087e-01	 1.5154868e+00	 1.2672227e-01	[ 1.2898173e+00]	 1.9768715e-01
      99	 1.4656244e+00	 1.2772235e-01	 1.5182004e+00	 1.2673948e-01	[ 1.2905974e+00]	 1.9018817e-01


.. parsed-literal::

     100	 1.4688527e+00	 1.2748924e-01	 1.5215140e+00	 1.2677056e-01	  1.2867953e+00 	 2.0919847e-01


.. parsed-literal::

     101	 1.4714270e+00	 1.2740954e-01	 1.5242681e+00	 1.2683238e-01	  1.2810472e+00 	 2.0378089e-01


.. parsed-literal::

     102	 1.4740830e+00	 1.2729967e-01	 1.5268765e+00	 1.2671790e-01	  1.2807230e+00 	 2.1635509e-01


.. parsed-literal::

     103	 1.4757294e+00	 1.2726265e-01	 1.5285594e+00	 1.2655633e-01	  1.2808945e+00 	 2.1553659e-01
     104	 1.4788953e+00	 1.2714828e-01	 1.5317767e+00	 1.2635987e-01	  1.2807720e+00 	 1.8935513e-01


.. parsed-literal::

     105	 1.4809589e+00	 1.2716691e-01	 1.5340065e+00	 1.2628918e-01	  1.2735177e+00 	 1.8635201e-01
     106	 1.4843928e+00	 1.2692093e-01	 1.5373529e+00	 1.2623269e-01	  1.2748260e+00 	 1.8123960e-01


.. parsed-literal::

     107	 1.4855388e+00	 1.2691601e-01	 1.5384573e+00	 1.2625057e-01	  1.2736127e+00 	 2.1475363e-01


.. parsed-literal::

     108	 1.4874995e+00	 1.2686016e-01	 1.5404283e+00	 1.2629797e-01	  1.2670127e+00 	 2.0371604e-01


.. parsed-literal::

     109	 1.4899930e+00	 1.2700494e-01	 1.5430070e+00	 1.2632589e-01	  1.2478504e+00 	 2.0458484e-01


.. parsed-literal::

     110	 1.4925779e+00	 1.2696006e-01	 1.5456377e+00	 1.2641066e-01	  1.2378737e+00 	 2.1203423e-01


.. parsed-literal::

     111	 1.4950710e+00	 1.2699842e-01	 1.5482210e+00	 1.2651944e-01	  1.2265106e+00 	 2.0368433e-01


.. parsed-literal::

     112	 1.4971421e+00	 1.2703368e-01	 1.5503413e+00	 1.2658673e-01	  1.2179273e+00 	 2.0391273e-01
     113	 1.4991796e+00	 1.2700097e-01	 1.5525360e+00	 1.2666615e-01	  1.2039750e+00 	 1.9719648e-01


.. parsed-literal::

     114	 1.5018782e+00	 1.2696817e-01	 1.5551791e+00	 1.2657842e-01	  1.2057667e+00 	 2.1124578e-01
     115	 1.5031164e+00	 1.2684956e-01	 1.5563768e+00	 1.2650378e-01	  1.2104439e+00 	 1.7681861e-01


.. parsed-literal::

     116	 1.5049690e+00	 1.2673048e-01	 1.5582478e+00	 1.2644232e-01	  1.2086962e+00 	 2.0930934e-01
     117	 1.5068401e+00	 1.2662895e-01	 1.5602310e+00	 1.2661945e-01	  1.2056925e+00 	 1.7502689e-01


.. parsed-literal::

     118	 1.5088154e+00	 1.2663155e-01	 1.5622418e+00	 1.2656712e-01	  1.1939141e+00 	 2.0591569e-01
     119	 1.5100092e+00	 1.2670326e-01	 1.5634464e+00	 1.2663135e-01	  1.1884643e+00 	 1.7994261e-01


.. parsed-literal::

     120	 1.5119087e+00	 1.2676017e-01	 1.5653766e+00	 1.2675694e-01	  1.1808606e+00 	 2.0858765e-01


.. parsed-literal::

     121	 1.5123286e+00	 1.2675538e-01	 1.5660000e+00	 1.2700932e-01	  1.1784239e+00 	 2.1823883e-01


.. parsed-literal::

     122	 1.5150326e+00	 1.2663034e-01	 1.5685328e+00	 1.2696581e-01	  1.1774144e+00 	 2.0567060e-01


.. parsed-literal::

     123	 1.5160538e+00	 1.2653166e-01	 1.5695075e+00	 1.2690030e-01	  1.1784451e+00 	 2.0226264e-01


.. parsed-literal::

     124	 1.5174034e+00	 1.2638814e-01	 1.5708534e+00	 1.2691924e-01	  1.1775598e+00 	 2.0857716e-01
     125	 1.5183242e+00	 1.2633184e-01	 1.5718569e+00	 1.2673300e-01	  1.1661552e+00 	 1.7977166e-01


.. parsed-literal::

     126	 1.5200573e+00	 1.2627033e-01	 1.5735639e+00	 1.2686960e-01	  1.1670355e+00 	 2.0645809e-01


.. parsed-literal::

     127	 1.5213470e+00	 1.2628736e-01	 1.5748825e+00	 1.2699720e-01	  1.1618227e+00 	 2.0975471e-01


.. parsed-literal::

     128	 1.5224154e+00	 1.2632266e-01	 1.5759830e+00	 1.2702395e-01	  1.1559133e+00 	 2.1066093e-01


.. parsed-literal::

     129	 1.5245185e+00	 1.2636713e-01	 1.5781371e+00	 1.2690799e-01	  1.1490586e+00 	 2.0826101e-01


.. parsed-literal::

     130	 1.5255922e+00	 1.2636529e-01	 1.5792581e+00	 1.2681682e-01	  1.1414412e+00 	 3.2248187e-01
     131	 1.5270747e+00	 1.2636614e-01	 1.5807388e+00	 1.2660663e-01	  1.1409648e+00 	 1.6922045e-01


.. parsed-literal::

     132	 1.5282819e+00	 1.2632407e-01	 1.5819494e+00	 1.2634102e-01	  1.1388927e+00 	 2.1018648e-01


.. parsed-literal::

     133	 1.5296696e+00	 1.2624954e-01	 1.5833483e+00	 1.2609114e-01	  1.1379234e+00 	 2.0961642e-01


.. parsed-literal::

     134	 1.5310750e+00	 1.2617694e-01	 1.5847641e+00	 1.2589101e-01	  1.1335891e+00 	 2.1725392e-01


.. parsed-literal::

     135	 1.5326603e+00	 1.2607561e-01	 1.5863878e+00	 1.2575208e-01	  1.1258158e+00 	 2.1255279e-01
     136	 1.5339250e+00	 1.2605566e-01	 1.5877060e+00	 1.2568672e-01	  1.1180909e+00 	 1.7903399e-01


.. parsed-literal::

     137	 1.5352149e+00	 1.2599625e-01	 1.5890070e+00	 1.2562497e-01	  1.1134537e+00 	 1.8614388e-01
     138	 1.5367636e+00	 1.2591645e-01	 1.5905926e+00	 1.2538657e-01	  1.1065619e+00 	 1.7932415e-01


.. parsed-literal::

     139	 1.5382916e+00	 1.2578995e-01	 1.5921644e+00	 1.2517346e-01	  1.1003901e+00 	 2.0429111e-01


.. parsed-literal::

     140	 1.5399930e+00	 1.2552644e-01	 1.5939058e+00	 1.2462572e-01	  1.0874864e+00 	 2.1093535e-01


.. parsed-literal::

     141	 1.5413062e+00	 1.2544827e-01	 1.5951698e+00	 1.2462636e-01	  1.0901834e+00 	 2.0688725e-01


.. parsed-literal::

     142	 1.5423891e+00	 1.2532770e-01	 1.5962525e+00	 1.2459256e-01	  1.0878017e+00 	 2.7980781e-01


.. parsed-literal::

     143	 1.5433671e+00	 1.2523542e-01	 1.5972605e+00	 1.2454407e-01	  1.0874912e+00 	 2.1409106e-01


.. parsed-literal::

     144	 1.5449846e+00	 1.2508405e-01	 1.5989959e+00	 1.2444991e-01	  1.0768116e+00 	 2.0774221e-01
     145	 1.5464900e+00	 1.2503088e-01	 1.6005530e+00	 1.2440005e-01	  1.0746300e+00 	 1.9746041e-01


.. parsed-literal::

     146	 1.5478812e+00	 1.2500894e-01	 1.6019997e+00	 1.2435987e-01	  1.0692009e+00 	 2.0349050e-01
     147	 1.5484219e+00	 1.2501312e-01	 1.6026534e+00	 1.2435936e-01	  1.0641658e+00 	 1.8623161e-01


.. parsed-literal::

     148	 1.5495491e+00	 1.2499036e-01	 1.6037266e+00	 1.2437908e-01	  1.0626537e+00 	 1.9174576e-01


.. parsed-literal::

     149	 1.5503194e+00	 1.2495357e-01	 1.6044841e+00	 1.2440183e-01	  1.0596708e+00 	 2.0515490e-01
     150	 1.5514065e+00	 1.2489137e-01	 1.6055916e+00	 1.2440547e-01	  1.0528582e+00 	 1.7882800e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 2s, sys: 1.05 s, total: 2min 3s
    Wall time: 31 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fe7446a0bb0>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.72 s, sys: 55 ms, total: 1.78 s
    Wall time: 542 ms


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

