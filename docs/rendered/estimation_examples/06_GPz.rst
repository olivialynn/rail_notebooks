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
       1	-3.3544947e-01	 3.1812042e-01	-3.2566267e-01	 3.2961921e-01	[-3.4830325e-01]	 4.6680260e-01


.. parsed-literal::

       2	-2.6112368e-01	 3.0570817e-01	-2.3474177e-01	 3.1816594e-01	[-2.7454862e-01]	 2.3523092e-01


.. parsed-literal::

       3	-2.1610009e-01	 2.8492680e-01	-1.7112990e-01	 3.0016793e-01	[-2.3853258e-01]	 2.7931881e-01


.. parsed-literal::

       4	-1.7507406e-01	 2.6701774e-01	-1.2432482e-01	 2.8591712e-01	[-2.2155785e-01]	 3.0107427e-01
       5	-1.2109127e-01	 2.5391537e-01	-9.0362888e-02	 2.7169383e-01	[-1.9384249e-01]	 2.0012522e-01


.. parsed-literal::

       6	-5.9638662e-02	 2.4973337e-01	-3.1565342e-02	 2.6676065e-01	[-9.9859771e-02]	 2.1224904e-01


.. parsed-literal::

       7	-4.2439523e-02	 2.4612056e-01	-1.8205690e-02	 2.6237408e-01	[-8.1218198e-02]	 2.1881604e-01


.. parsed-literal::

       8	-3.0965251e-02	 2.4437347e-01	-1.0123392e-02	 2.6012820e-01	[-7.4807556e-02]	 2.1160221e-01


.. parsed-literal::

       9	-1.5424694e-02	 2.4148243e-01	 2.0836300e-03	 2.5629240e-01	[-5.8393529e-02]	 2.1146154e-01


.. parsed-literal::

      10	-4.8561252e-03	 2.3952313e-01	 1.0634938e-02	 2.5360337e-01	[-5.1858147e-02]	 2.0914292e-01
      11	-2.9608783e-04	 2.3876090e-01	 1.4477677e-02	 2.5213312e-01	[-4.4097028e-02]	 1.8880677e-01


.. parsed-literal::

      12	 4.2538471e-03	 2.3798705e-01	 1.8412719e-02	 2.5118265e-01	[-3.7945854e-02]	 2.0939875e-01
      13	 8.6444787e-03	 2.3709422e-01	 2.2912764e-02	 2.5084053e-01	[-3.5147168e-02]	 1.9788718e-01


.. parsed-literal::

      14	 9.9194798e-02	 2.2408459e-01	 1.1865680e-01	 2.3888948e-01	[ 6.9563302e-02]	 3.2731891e-01


.. parsed-literal::

      15	 1.3165389e-01	 2.2225610e-01	 1.5401621e-01	 2.4103539e-01	[ 1.1979342e-01]	 2.0799708e-01


.. parsed-literal::

      16	 1.9839070e-01	 2.1651107e-01	 2.2219848e-01	 2.3651362e-01	[ 1.8815766e-01]	 2.1024776e-01
      17	 2.9779932e-01	 2.1051590e-01	 3.2935871e-01	 2.2514054e-01	[ 2.8910801e-01]	 1.9675779e-01


.. parsed-literal::

      18	 3.3945455e-01	 2.0675876e-01	 3.7214645e-01	 2.2174783e-01	[ 3.3434818e-01]	 2.0102620e-01


.. parsed-literal::

      19	 3.9605096e-01	 2.0379374e-01	 4.2891670e-01	 2.1980987e-01	[ 3.9677161e-01]	 2.1253514e-01


.. parsed-literal::

      20	 4.6626289e-01	 2.0138024e-01	 4.9983880e-01	 2.2002438e-01	[ 4.7290280e-01]	 2.0877218e-01
      21	 5.6319738e-01	 1.9666598e-01	 5.9834351e-01	 2.1254236e-01	[ 5.9957053e-01]	 1.9556642e-01


.. parsed-literal::

      22	 6.1156459e-01	 1.9724317e-01	 6.5235761e-01	 2.1186034e-01	[ 7.0598177e-01]	 2.0411372e-01
      23	 6.6584888e-01	 1.9587830e-01	 7.0336188e-01	 2.1073974e-01	[ 7.3894494e-01]	 1.7211199e-01


.. parsed-literal::

      24	 6.8717576e-01	 1.9107565e-01	 7.2447592e-01	 2.0630271e-01	[ 7.5821836e-01]	 2.1846581e-01
      25	 7.0860287e-01	 1.8864672e-01	 7.4576504e-01	 2.0319050e-01	[ 7.7609272e-01]	 1.9313288e-01


.. parsed-literal::

      26	 7.3029502e-01	 1.8917636e-01	 7.6743945e-01	 2.0368786e-01	[ 7.9539589e-01]	 2.0700788e-01


.. parsed-literal::

      27	 7.5662012e-01	 1.8817955e-01	 7.9459614e-01	 2.0367461e-01	[ 8.2051033e-01]	 2.0993352e-01


.. parsed-literal::

      28	 7.9226468e-01	 1.8816062e-01	 8.3130009e-01	 2.0444253e-01	[ 8.5937155e-01]	 2.1460009e-01
      29	 8.1439523e-01	 1.9182199e-01	 8.5376587e-01	 2.0995666e-01	[ 8.7117560e-01]	 1.7902231e-01


.. parsed-literal::

      30	 8.3914314e-01	 1.8837373e-01	 8.7888508e-01	 2.0868532e-01	[ 8.9484608e-01]	 2.0177078e-01
      31	 8.5902891e-01	 1.8604779e-01	 8.9891909e-01	 2.0628463e-01	[ 9.1412222e-01]	 1.8501306e-01


.. parsed-literal::

      32	 8.8628320e-01	 1.8205868e-01	 9.2742485e-01	 2.0274782e-01	[ 9.3766384e-01]	 1.8778682e-01


.. parsed-literal::

      33	 9.0470531e-01	 1.7976272e-01	 9.4660735e-01	 2.0016939e-01	[ 9.5650904e-01]	 2.0664477e-01


.. parsed-literal::

      34	 9.1933600e-01	 1.7722284e-01	 9.6128145e-01	 1.9767070e-01	[ 9.6511768e-01]	 2.0746279e-01
      35	 9.3262023e-01	 1.7471659e-01	 9.7456761e-01	 1.9498008e-01	[ 9.7804972e-01]	 1.7437506e-01


.. parsed-literal::

      36	 9.4672843e-01	 1.7191481e-01	 9.8941327e-01	 1.9149146e-01	[ 9.8086583e-01]	 2.1473193e-01


.. parsed-literal::

      37	 9.6779840e-01	 1.6956141e-01	 1.0106335e+00	 1.8867938e-01	[ 1.0038298e+00]	 2.0963907e-01


.. parsed-literal::

      38	 9.8443596e-01	 1.6699983e-01	 1.0280316e+00	 1.8589127e-01	[ 1.0210495e+00]	 2.0725536e-01


.. parsed-literal::

      39	 1.0054489e+00	 1.6453171e-01	 1.0502512e+00	 1.8224473e-01	[ 1.0417939e+00]	 2.0455980e-01


.. parsed-literal::

      40	 1.0199614e+00	 1.6203712e-01	 1.0654990e+00	 1.7953583e-01	[ 1.0542542e+00]	 2.0985985e-01


.. parsed-literal::

      41	 1.0329750e+00	 1.6082564e-01	 1.0787709e+00	 1.7846136e-01	[ 1.0659357e+00]	 2.1610308e-01


.. parsed-literal::

      42	 1.0442931e+00	 1.5892745e-01	 1.0903410e+00	 1.7709530e-01	[ 1.0719705e+00]	 2.0131540e-01


.. parsed-literal::

      43	 1.0563463e+00	 1.5813954e-01	 1.1024897e+00	 1.7660951e-01	[ 1.0848450e+00]	 2.0728278e-01


.. parsed-literal::

      44	 1.0669113e+00	 1.5683624e-01	 1.1134457e+00	 1.7537655e-01	[ 1.0900101e+00]	 2.1381807e-01
      45	 1.0801069e+00	 1.5585846e-01	 1.1266680e+00	 1.7476037e-01	[ 1.1058330e+00]	 1.9997215e-01


.. parsed-literal::

      46	 1.0951905e+00	 1.5357293e-01	 1.1419293e+00	 1.7264223e-01	[ 1.1156258e+00]	 2.1037006e-01


.. parsed-literal::

      47	 1.1123677e+00	 1.5010041e-01	 1.1593180e+00	 1.6868083e-01	[ 1.1340694e+00]	 2.0672989e-01
      48	 1.1237234e+00	 1.4801983e-01	 1.1704104e+00	 1.6629057e-01	[ 1.1453348e+00]	 1.7402506e-01


.. parsed-literal::

      49	 1.1331262e+00	 1.4679891e-01	 1.1798254e+00	 1.6506309e-01	[ 1.1549527e+00]	 2.1305346e-01


.. parsed-literal::

      50	 1.1475236e+00	 1.4467857e-01	 1.1950534e+00	 1.6260028e-01	[ 1.1669724e+00]	 2.1121740e-01


.. parsed-literal::

      51	 1.1579971e+00	 1.4383893e-01	 1.2057964e+00	 1.6259602e-01	[ 1.1780391e+00]	 2.1611977e-01


.. parsed-literal::

      52	 1.1704215e+00	 1.4281473e-01	 1.2183983e+00	 1.6152669e-01	[ 1.1865353e+00]	 2.0652580e-01
      53	 1.1825635e+00	 1.4246803e-01	 1.2309229e+00	 1.6144840e-01	[ 1.1909111e+00]	 1.9893909e-01


.. parsed-literal::

      54	 1.1939393e+00	 1.4207353e-01	 1.2428216e+00	 1.6081546e-01	[ 1.1939386e+00]	 1.9941425e-01
      55	 1.2056964e+00	 1.4175716e-01	 1.2547401e+00	 1.6080582e-01	[ 1.1950428e+00]	 2.0058489e-01


.. parsed-literal::

      56	 1.2159840e+00	 1.4092366e-01	 1.2651475e+00	 1.5948860e-01	[ 1.2010730e+00]	 2.0470262e-01


.. parsed-literal::

      57	 1.2259864e+00	 1.4027192e-01	 1.2753103e+00	 1.5868851e-01	  1.2007026e+00 	 2.1006513e-01


.. parsed-literal::

      58	 1.2348036e+00	 1.3959705e-01	 1.2843031e+00	 1.5838656e-01	  1.1994081e+00 	 2.1785784e-01


.. parsed-literal::

      59	 1.2422447e+00	 1.3900936e-01	 1.2919063e+00	 1.5843765e-01	  1.1993808e+00 	 2.1054173e-01


.. parsed-literal::

      60	 1.2513042e+00	 1.3821245e-01	 1.3011560e+00	 1.5859300e-01	  1.1977140e+00 	 2.0648289e-01
      61	 1.2585054e+00	 1.3726489e-01	 1.3088312e+00	 1.5898313e-01	  1.1937709e+00 	 1.7226028e-01


.. parsed-literal::

      62	 1.2674971e+00	 1.3679044e-01	 1.3175906e+00	 1.5899600e-01	[ 1.2069058e+00]	 1.6917443e-01
      63	 1.2727562e+00	 1.3647614e-01	 1.3229338e+00	 1.5871263e-01	[ 1.2109657e+00]	 1.8375564e-01


.. parsed-literal::

      64	 1.2796220e+00	 1.3597807e-01	 1.3298811e+00	 1.5848992e-01	[ 1.2184528e+00]	 2.1432018e-01


.. parsed-literal::

      65	 1.2871353e+00	 1.3496785e-01	 1.3379369e+00	 1.5727713e-01	[ 1.2188848e+00]	 2.1378756e-01


.. parsed-literal::

      66	 1.2974197e+00	 1.3449390e-01	 1.3481965e+00	 1.5710841e-01	[ 1.2306718e+00]	 2.1581578e-01
      67	 1.3032810e+00	 1.3418520e-01	 1.3541433e+00	 1.5680008e-01	[ 1.2353128e+00]	 1.9762421e-01


.. parsed-literal::

      68	 1.3108072e+00	 1.3381752e-01	 1.3619502e+00	 1.5649130e-01	[ 1.2371958e+00]	 1.7971277e-01


.. parsed-literal::

      69	 1.3145115e+00	 1.3373014e-01	 1.3660558e+00	 1.5632087e-01	[ 1.2376311e+00]	 2.0712590e-01


.. parsed-literal::

      70	 1.3220067e+00	 1.3339583e-01	 1.3732399e+00	 1.5617768e-01	[ 1.2452659e+00]	 2.1937990e-01
      71	 1.3267869e+00	 1.3304682e-01	 1.3780376e+00	 1.5589294e-01	[ 1.2481354e+00]	 1.9961691e-01


.. parsed-literal::

      72	 1.3325047e+00	 1.3259352e-01	 1.3837037e+00	 1.5542002e-01	[ 1.2537643e+00]	 2.0203137e-01
      73	 1.3400795e+00	 1.3202692e-01	 1.3914793e+00	 1.5423952e-01	[ 1.2539098e+00]	 1.8853498e-01


.. parsed-literal::

      74	 1.3465987e+00	 1.3138510e-01	 1.3978471e+00	 1.5339334e-01	[ 1.2644705e+00]	 1.9762921e-01
      75	 1.3504289e+00	 1.3114852e-01	 1.4015659e+00	 1.5313448e-01	[ 1.2684451e+00]	 1.9968438e-01


.. parsed-literal::

      76	 1.3564821e+00	 1.3061995e-01	 1.4077360e+00	 1.5230788e-01	[ 1.2733542e+00]	 2.0270467e-01


.. parsed-literal::

      77	 1.3636051e+00	 1.2979444e-01	 1.4149980e+00	 1.5131525e-01	[ 1.2841411e+00]	 2.0853829e-01
      78	 1.3683454e+00	 1.2918029e-01	 1.4202799e+00	 1.5032885e-01	[ 1.2843008e+00]	 1.9081330e-01


.. parsed-literal::

      79	 1.3753786e+00	 1.2871006e-01	 1.4270611e+00	 1.5005308e-01	[ 1.2927215e+00]	 2.0469093e-01
      80	 1.3795157e+00	 1.2841660e-01	 1.4311513e+00	 1.4994311e-01	[ 1.2955028e+00]	 1.9872594e-01


.. parsed-literal::

      81	 1.3865035e+00	 1.2778910e-01	 1.4383038e+00	 1.4943635e-01	[ 1.2978275e+00]	 2.0563364e-01
      82	 1.3927715e+00	 1.2710352e-01	 1.4446426e+00	 1.4936160e-01	[ 1.2984844e+00]	 1.9790888e-01


.. parsed-literal::

      83	 1.3990794e+00	 1.2666283e-01	 1.4510502e+00	 1.4869257e-01	[ 1.3020291e+00]	 2.1757817e-01


.. parsed-literal::

      84	 1.4035739e+00	 1.2657225e-01	 1.4553672e+00	 1.4854811e-01	[ 1.3058841e+00]	 2.0975089e-01
      85	 1.4099930e+00	 1.2633820e-01	 1.4618982e+00	 1.4822065e-01	[ 1.3067814e+00]	 1.8316197e-01


.. parsed-literal::

      86	 1.4148928e+00	 1.2605859e-01	 1.4670153e+00	 1.4790914e-01	  1.3051892e+00 	 1.8575597e-01
      87	 1.4195317e+00	 1.2570974e-01	 1.4716974e+00	 1.4779706e-01	[ 1.3079193e+00]	 1.9814634e-01


.. parsed-literal::

      88	 1.4238998e+00	 1.2533552e-01	 1.4761440e+00	 1.4761939e-01	  1.3072486e+00 	 2.1563888e-01


.. parsed-literal::

      89	 1.4282951e+00	 1.2502962e-01	 1.4806164e+00	 1.4762558e-01	[ 1.3086966e+00]	 2.1272397e-01


.. parsed-literal::

      90	 1.4324169e+00	 1.2477171e-01	 1.4847635e+00	 1.4755963e-01	  1.3070942e+00 	 2.1211123e-01


.. parsed-literal::

      91	 1.4361059e+00	 1.2462046e-01	 1.4885867e+00	 1.4767685e-01	  1.3021807e+00 	 2.1705008e-01


.. parsed-literal::

      92	 1.4389912e+00	 1.2453716e-01	 1.4915385e+00	 1.4751448e-01	  1.3034094e+00 	 2.0973063e-01
      93	 1.4429031e+00	 1.2444028e-01	 1.4955142e+00	 1.4752132e-01	  1.3042932e+00 	 2.0063353e-01


.. parsed-literal::

      94	 1.4466386e+00	 1.2451231e-01	 1.4993716e+00	 1.4778954e-01	  1.2977633e+00 	 2.0669603e-01


.. parsed-literal::

      95	 1.4493116e+00	 1.2422309e-01	 1.5021520e+00	 1.4759385e-01	  1.3001654e+00 	 2.0481920e-01


.. parsed-literal::

      96	 1.4518231e+00	 1.2408592e-01	 1.5046353e+00	 1.4753876e-01	  1.3002366e+00 	 2.1179318e-01
      97	 1.4550548e+00	 1.2387726e-01	 1.5079876e+00	 1.4759676e-01	  1.2951066e+00 	 1.9227767e-01


.. parsed-literal::

      98	 1.4583718e+00	 1.2365011e-01	 1.5114337e+00	 1.4748888e-01	  1.2917803e+00 	 1.8559909e-01


.. parsed-literal::

      99	 1.4616878e+00	 1.2338899e-01	 1.5150857e+00	 1.4790303e-01	  1.2848437e+00 	 2.0581555e-01


.. parsed-literal::

     100	 1.4662333e+00	 1.2329196e-01	 1.5195437e+00	 1.4758894e-01	  1.2874504e+00 	 2.1021366e-01
     101	 1.4682159e+00	 1.2324489e-01	 1.5214183e+00	 1.4760844e-01	  1.2909501e+00 	 1.9876361e-01


.. parsed-literal::

     102	 1.4715132e+00	 1.2318204e-01	 1.5246904e+00	 1.4780275e-01	  1.2917586e+00 	 2.1117759e-01
     103	 1.4736535e+00	 1.2306732e-01	 1.5268480e+00	 1.4853792e-01	  1.2884448e+00 	 1.9502878e-01


.. parsed-literal::

     104	 1.4771189e+00	 1.2300560e-01	 1.5302637e+00	 1.4844149e-01	  1.2893362e+00 	 2.0162272e-01
     105	 1.4797432e+00	 1.2288738e-01	 1.5329258e+00	 1.4843134e-01	  1.2887955e+00 	 1.9906950e-01


.. parsed-literal::

     106	 1.4818762e+00	 1.2279266e-01	 1.5350745e+00	 1.4847548e-01	  1.2900816e+00 	 2.0042896e-01


.. parsed-literal::

     107	 1.4860399e+00	 1.2260460e-01	 1.5392685e+00	 1.4848471e-01	  1.2900904e+00 	 2.0365763e-01


.. parsed-literal::

     108	 1.4881407e+00	 1.2244958e-01	 1.5414150e+00	 1.4871386e-01	  1.2893242e+00 	 3.1973886e-01
     109	 1.4902152e+00	 1.2231079e-01	 1.5434775e+00	 1.4863929e-01	  1.2918172e+00 	 1.9960713e-01


.. parsed-literal::

     110	 1.4920640e+00	 1.2222588e-01	 1.5453133e+00	 1.4860770e-01	  1.2909988e+00 	 2.0968199e-01
     111	 1.4945290e+00	 1.2194641e-01	 1.5478240e+00	 1.4848445e-01	  1.2928755e+00 	 1.8939853e-01


.. parsed-literal::

     112	 1.4974256e+00	 1.2183267e-01	 1.5507843e+00	 1.4845146e-01	  1.2881373e+00 	 2.1563601e-01


.. parsed-literal::

     113	 1.4996453e+00	 1.2165052e-01	 1.5530423e+00	 1.4846244e-01	  1.2888552e+00 	 2.1621013e-01


.. parsed-literal::

     114	 1.5023472e+00	 1.2146326e-01	 1.5558246e+00	 1.4828892e-01	  1.2954738e+00 	 2.1300316e-01


.. parsed-literal::

     115	 1.5046991e+00	 1.2125726e-01	 1.5582928e+00	 1.4827318e-01	  1.2951369e+00 	 2.1077657e-01
     116	 1.5069957e+00	 1.2109954e-01	 1.5606152e+00	 1.4795984e-01	  1.3006298e+00 	 1.9574356e-01


.. parsed-literal::

     117	 1.5090169e+00	 1.2096247e-01	 1.5626839e+00	 1.4773766e-01	  1.3031589e+00 	 1.7696619e-01
     118	 1.5109217e+00	 1.2085080e-01	 1.5646497e+00	 1.4757083e-01	  1.3018450e+00 	 1.9624543e-01


.. parsed-literal::

     119	 1.5136244e+00	 1.2057885e-01	 1.5675345e+00	 1.4769196e-01	  1.2956878e+00 	 2.0260596e-01
     120	 1.5158885e+00	 1.2049575e-01	 1.5698293e+00	 1.4742504e-01	  1.2932250e+00 	 1.7506552e-01


.. parsed-literal::

     121	 1.5172343e+00	 1.2044068e-01	 1.5711179e+00	 1.4750528e-01	  1.2928041e+00 	 1.8698239e-01
     122	 1.5192913e+00	 1.2037891e-01	 1.5731593e+00	 1.4776902e-01	  1.2913604e+00 	 2.0128989e-01


.. parsed-literal::

     123	 1.5209953e+00	 1.2043001e-01	 1.5749448e+00	 1.4829334e-01	  1.2892734e+00 	 2.0923758e-01
     124	 1.5231048e+00	 1.2042235e-01	 1.5770154e+00	 1.4848981e-01	  1.2908022e+00 	 1.8342400e-01


.. parsed-literal::

     125	 1.5244541e+00	 1.2037387e-01	 1.5783864e+00	 1.4859437e-01	  1.2901767e+00 	 1.8105292e-01
     126	 1.5258216e+00	 1.2035957e-01	 1.5797901e+00	 1.4877051e-01	  1.2893765e+00 	 2.0520449e-01


.. parsed-literal::

     127	 1.5276283e+00	 1.2033300e-01	 1.5817027e+00	 1.4922810e-01	  1.2789516e+00 	 2.1932268e-01


.. parsed-literal::

     128	 1.5293412e+00	 1.2034281e-01	 1.5834808e+00	 1.4940115e-01	  1.2780415e+00 	 2.1025205e-01
     129	 1.5306309e+00	 1.2034792e-01	 1.5847899e+00	 1.4959246e-01	  1.2778970e+00 	 1.9709945e-01


.. parsed-literal::

     130	 1.5321144e+00	 1.2027383e-01	 1.5862799e+00	 1.4958734e-01	  1.2769299e+00 	 2.0094442e-01


.. parsed-literal::

     131	 1.5340051e+00	 1.2020930e-01	 1.5881832e+00	 1.4974952e-01	  1.2788213e+00 	 2.0961809e-01


.. parsed-literal::

     132	 1.5358247e+00	 1.2012422e-01	 1.5900197e+00	 1.4980564e-01	  1.2800459e+00 	 2.1235275e-01


.. parsed-literal::

     133	 1.5371379e+00	 1.1999633e-01	 1.5913207e+00	 1.4970091e-01	  1.2799208e+00 	 2.1266127e-01


.. parsed-literal::

     134	 1.5383955e+00	 1.1993260e-01	 1.5925555e+00	 1.4960158e-01	  1.2798035e+00 	 2.1807766e-01


.. parsed-literal::

     135	 1.5404450e+00	 1.1979812e-01	 1.5946090e+00	 1.4947145e-01	  1.2768829e+00 	 2.1115470e-01


.. parsed-literal::

     136	 1.5423950e+00	 1.1963789e-01	 1.5965950e+00	 1.4924291e-01	  1.2740425e+00 	 2.1662164e-01


.. parsed-literal::

     137	 1.5433885e+00	 1.1955092e-01	 1.5977725e+00	 1.4940504e-01	  1.2649487e+00 	 2.1484399e-01
     138	 1.5457073e+00	 1.1949051e-01	 1.5999927e+00	 1.4923283e-01	  1.2677329e+00 	 2.0364237e-01


.. parsed-literal::

     139	 1.5465702e+00	 1.1946744e-01	 1.6008401e+00	 1.4916009e-01	  1.2702783e+00 	 2.1171927e-01


.. parsed-literal::

     140	 1.5478412e+00	 1.1943883e-01	 1.6021306e+00	 1.4922468e-01	  1.2694134e+00 	 2.0411253e-01
     141	 1.5489413e+00	 1.1933587e-01	 1.6033871e+00	 1.4911010e-01	  1.2667156e+00 	 1.9205236e-01


.. parsed-literal::

     142	 1.5507338e+00	 1.1928955e-01	 1.6051157e+00	 1.4932083e-01	  1.2640944e+00 	 1.7909861e-01


.. parsed-literal::

     143	 1.5516678e+00	 1.1920844e-01	 1.6060462e+00	 1.4936045e-01	  1.2607829e+00 	 2.0804358e-01


.. parsed-literal::

     144	 1.5527304e+00	 1.1908112e-01	 1.6071242e+00	 1.4929274e-01	  1.2584970e+00 	 2.0929337e-01
     145	 1.5543715e+00	 1.1883510e-01	 1.6087927e+00	 1.4907198e-01	  1.2573712e+00 	 2.0119143e-01


.. parsed-literal::

     146	 1.5551502e+00	 1.1854447e-01	 1.6096311e+00	 1.4865238e-01	  1.2576139e+00 	 1.7697382e-01
     147	 1.5568485e+00	 1.1855320e-01	 1.6112436e+00	 1.4862959e-01	  1.2605551e+00 	 1.8049932e-01


.. parsed-literal::

     148	 1.5575926e+00	 1.1853120e-01	 1.6119539e+00	 1.4855778e-01	  1.2632297e+00 	 2.0162129e-01
     149	 1.5585262e+00	 1.1844243e-01	 1.6128762e+00	 1.4840827e-01	  1.2647811e+00 	 2.0285153e-01


.. parsed-literal::

     150	 1.5600862e+00	 1.1824032e-01	 1.6144402e+00	 1.4813461e-01	  1.2643746e+00 	 2.0834684e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.06 s, total: 2min 4s
    Wall time: 31.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f63e4d05390>



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


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.02 s, sys: 57 ms, total: 2.08 s
    Wall time: 618 ms


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

