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
       1	-3.4150458e-01	 3.2011805e-01	-3.3183701e-01	 3.2276309e-01	[-3.3705716e-01]	 4.6582007e-01


.. parsed-literal::

       2	-2.7078747e-01	 3.0936437e-01	-2.4686918e-01	 3.1087806e-01	[-2.5172720e-01]	 2.3173404e-01


.. parsed-literal::

       3	-2.2536808e-01	 2.8830662e-01	-1.8276526e-01	 2.8713884e-01	[-1.8033483e-01]	 2.8550625e-01
       4	-1.9760708e-01	 2.6487427e-01	-1.5843286e-01	 2.6403650e-01	[-1.5411484e-01]	 1.8149424e-01


.. parsed-literal::

       5	-1.0166732e-01	 2.5668057e-01	-6.6712993e-02	 2.5580037e-01	[-6.3708248e-02]	 1.9031763e-01


.. parsed-literal::

       6	-6.9797834e-02	 2.5146344e-01	-3.8691950e-02	 2.5261832e-01	[-4.1451160e-02]	 2.0387411e-01


.. parsed-literal::

       7	-5.0624806e-02	 2.4853542e-01	-2.7077641e-02	 2.4865043e-01	[-2.6864296e-02]	 2.2052956e-01


.. parsed-literal::

       8	-3.9811900e-02	 2.4678666e-01	-1.9748432e-02	 2.4694781e-01	[-1.9972273e-02]	 2.0900512e-01


.. parsed-literal::

       9	-2.6090431e-02	 2.4425787e-01	-8.9219232e-03	 2.4466489e-01	[-1.0391407e-02]	 2.1235538e-01


.. parsed-literal::

      10	-1.8210435e-02	 2.4297456e-01	-3.0603175e-03	 2.4226040e-01	[-2.2213869e-04]	 2.1344280e-01


.. parsed-literal::

      11	-1.2537808e-02	 2.4185925e-01	 2.0248949e-03	 2.4116840e-01	[ 4.6914378e-03]	 2.1690154e-01


.. parsed-literal::

      12	-1.0366065e-02	 2.4142965e-01	 4.0067215e-03	 2.4072039e-01	[ 6.7976164e-03]	 2.1916270e-01
      13	-4.3270025e-03	 2.4023959e-01	 9.7229603e-03	 2.3940099e-01	[ 1.3047299e-02]	 1.9443274e-01


.. parsed-literal::

      14	 6.8688799e-03	 2.3754118e-01	 2.3789634e-02	 2.3599259e-01	[ 3.0315404e-02]	 1.8966246e-01
      15	 8.9197786e-02	 2.2549598e-01	 1.2028162e-01	 2.2773264e-01	[ 1.2776596e-01]	 1.8836260e-01


.. parsed-literal::

      16	 2.3224417e-01	 2.2035163e-01	 2.6629345e-01	 2.2341033e-01	[ 2.6384000e-01]	 2.1504354e-01


.. parsed-literal::

      17	 2.5494952e-01	 2.1393146e-01	 2.8636961e-01	 2.1526396e-01	[ 2.8554035e-01]	 2.0968866e-01


.. parsed-literal::

      18	 2.9236950e-01	 2.0841759e-01	 3.2505604e-01	 2.1058804e-01	[ 3.2399098e-01]	 2.0599818e-01


.. parsed-literal::

      19	 2.9272215e-01	 2.1221461e-01	 3.2678883e-01	 2.1617483e-01	[ 3.2514907e-01]	 2.1428466e-01


.. parsed-literal::

      20	 3.7243963e-01	 2.0276268e-01	 4.0649181e-01	 2.0636389e-01	[ 3.9985813e-01]	 2.1217561e-01


.. parsed-literal::

      21	 4.2734006e-01	 1.9638192e-01	 4.5952175e-01	 1.9876684e-01	[ 4.5556640e-01]	 2.1109223e-01
      22	 5.0372638e-01	 1.9320384e-01	 5.3691680e-01	 1.9603682e-01	[ 5.3553825e-01]	 1.7250776e-01


.. parsed-literal::

      23	 6.0390316e-01	 1.9049836e-01	 6.4140775e-01	 1.9189523e-01	[ 6.4649019e-01]	 2.2124219e-01


.. parsed-literal::

      24	 6.3552407e-01	 1.9248221e-01	 6.7473020e-01	 1.9251180e-01	[ 6.6805052e-01]	 2.1479344e-01


.. parsed-literal::

      25	 6.6425815e-01	 1.8872616e-01	 7.0251705e-01	 1.8979860e-01	[ 6.9764991e-01]	 2.0298839e-01


.. parsed-literal::

      26	 6.9438198e-01	 1.8703085e-01	 7.3266758e-01	 1.8887450e-01	[ 7.2722025e-01]	 2.1204758e-01
      27	 7.1632817e-01	 1.9040633e-01	 7.5424677e-01	 1.9089833e-01	[ 7.3878047e-01]	 2.0118690e-01


.. parsed-literal::

      28	 7.7444935e-01	 1.8503196e-01	 8.1258230e-01	 1.8344750e-01	[ 8.0285204e-01]	 2.0019603e-01


.. parsed-literal::

      29	 8.1011979e-01	 1.8447409e-01	 8.5023138e-01	 1.8178690e-01	[ 8.3656229e-01]	 2.0501328e-01


.. parsed-literal::

      30	 8.3718754e-01	 1.8513764e-01	 8.7707270e-01	 1.8240062e-01	[ 8.6517336e-01]	 2.0997071e-01
      31	 8.5770417e-01	 1.8524059e-01	 8.9949391e-01	 1.8424706e-01	[ 8.7256091e-01]	 1.9516730e-01


.. parsed-literal::

      32	 8.8732052e-01	 1.7851221e-01	 9.2941322e-01	 1.8011053e-01	[ 8.9968866e-01]	 2.0823884e-01


.. parsed-literal::

      33	 9.0054176e-01	 1.7648956e-01	 9.4199099e-01	 1.7848183e-01	[ 9.1313133e-01]	 2.1027970e-01


.. parsed-literal::

      34	 9.2529633e-01	 1.7103071e-01	 9.6719533e-01	 1.7497904e-01	[ 9.3580933e-01]	 2.1385050e-01


.. parsed-literal::

      35	 9.5214581e-01	 1.6759893e-01	 9.9509576e-01	 1.7294093e-01	[ 9.5371499e-01]	 2.1128249e-01


.. parsed-literal::

      36	 9.7242774e-01	 1.6462976e-01	 1.0160460e+00	 1.7100232e-01	[ 9.6790688e-01]	 2.1004152e-01
      37	 9.8945227e-01	 1.6321619e-01	 1.0334758e+00	 1.6905593e-01	[ 9.8719060e-01]	 1.8467164e-01


.. parsed-literal::

      38	 1.0091899e+00	 1.6417650e-01	 1.0543923e+00	 1.6949465e-01	[ 1.0064769e+00]	 2.1437693e-01
      39	 1.0261455e+00	 1.6271306e-01	 1.0715631e+00	 1.6762064e-01	[ 1.0272897e+00]	 1.7963505e-01


.. parsed-literal::

      40	 1.0371194e+00	 1.6227940e-01	 1.0826186e+00	 1.6734139e-01	[ 1.0395251e+00]	 2.1257901e-01


.. parsed-literal::

      41	 1.0513398e+00	 1.6167009e-01	 1.0972612e+00	 1.6629967e-01	[ 1.0569424e+00]	 2.0982218e-01


.. parsed-literal::

      42	 1.0621896e+00	 1.6029688e-01	 1.1085081e+00	 1.6534479e-01	[ 1.0723061e+00]	 2.1445537e-01


.. parsed-literal::

      43	 1.0736505e+00	 1.5912269e-01	 1.1202713e+00	 1.6405929e-01	[ 1.0875417e+00]	 2.1537089e-01


.. parsed-literal::

      44	 1.0845466e+00	 1.5681562e-01	 1.1315301e+00	 1.6248281e-01	[ 1.0988440e+00]	 2.1122646e-01


.. parsed-literal::

      45	 1.0989192e+00	 1.5390819e-01	 1.1463425e+00	 1.6126141e-01	[ 1.1151619e+00]	 2.1892309e-01


.. parsed-literal::

      46	 1.1054405e+00	 1.5263476e-01	 1.1529227e+00	 1.5963187e-01	[ 1.1283587e+00]	 2.1189666e-01


.. parsed-literal::

      47	 1.1129587e+00	 1.5202720e-01	 1.1601557e+00	 1.5941551e-01	[ 1.1311184e+00]	 2.1803308e-01


.. parsed-literal::

      48	 1.1184660e+00	 1.5134411e-01	 1.1657568e+00	 1.5897043e-01	[ 1.1344428e+00]	 2.0923400e-01
      49	 1.1289369e+00	 1.4986814e-01	 1.1764910e+00	 1.5790793e-01	[ 1.1411434e+00]	 1.8387318e-01


.. parsed-literal::

      50	 1.1483385e+00	 1.4734354e-01	 1.1960883e+00	 1.5622689e-01	[ 1.1557311e+00]	 1.9772530e-01


.. parsed-literal::

      51	 1.1585807e+00	 1.4650920e-01	 1.2067354e+00	 1.5601720e-01	[ 1.1638786e+00]	 3.2222176e-01
      52	 1.1709474e+00	 1.4525219e-01	 1.2190812e+00	 1.5540004e-01	[ 1.1684827e+00]	 1.9535065e-01


.. parsed-literal::

      53	 1.1862481e+00	 1.4365814e-01	 1.2341992e+00	 1.5543552e-01	[ 1.1699461e+00]	 2.0299888e-01
      54	 1.1982596e+00	 1.4217815e-01	 1.2465516e+00	 1.5486827e-01	[ 1.1767587e+00]	 1.8740392e-01


.. parsed-literal::

      55	 1.2084250e+00	 1.4152547e-01	 1.2566266e+00	 1.5472010e-01	[ 1.1837747e+00]	 2.1834874e-01
      56	 1.2174934e+00	 1.4104647e-01	 1.2659336e+00	 1.5470606e-01	[ 1.1922119e+00]	 1.8713641e-01


.. parsed-literal::

      57	 1.2272610e+00	 1.4122073e-01	 1.2760356e+00	 1.5447163e-01	[ 1.2089595e+00]	 2.0262337e-01


.. parsed-literal::

      58	 1.2367773e+00	 1.4165631e-01	 1.2859391e+00	 1.5450219e-01	[ 1.2216859e+00]	 2.1331191e-01
      59	 1.2458129e+00	 1.4109032e-01	 1.2950116e+00	 1.5409237e-01	[ 1.2300404e+00]	 2.0413733e-01


.. parsed-literal::

      60	 1.2545846e+00	 1.4057192e-01	 1.3040912e+00	 1.5383246e-01	[ 1.2395725e+00]	 2.1040392e-01
      61	 1.2631546e+00	 1.4040382e-01	 1.3129701e+00	 1.5388509e-01	[ 1.2452100e+00]	 1.9523406e-01


.. parsed-literal::

      62	 1.2726546e+00	 1.3927489e-01	 1.3225944e+00	 1.5315594e-01	[ 1.2595478e+00]	 1.9987202e-01


.. parsed-literal::

      63	 1.2803019e+00	 1.3855506e-01	 1.3302467e+00	 1.5273764e-01	[ 1.2656385e+00]	 2.0642447e-01


.. parsed-literal::

      64	 1.2940250e+00	 1.3627604e-01	 1.3441486e+00	 1.5057777e-01	[ 1.2733341e+00]	 2.1180534e-01


.. parsed-literal::

      65	 1.2999152e+00	 1.3509079e-01	 1.3501458e+00	 1.5012020e-01	[ 1.2831457e+00]	 2.0950007e-01
      66	 1.3103643e+00	 1.3378349e-01	 1.3603613e+00	 1.4895407e-01	[ 1.2880278e+00]	 2.0680737e-01


.. parsed-literal::

      67	 1.3159364e+00	 1.3347275e-01	 1.3660885e+00	 1.4855786e-01	[ 1.2892341e+00]	 2.1679354e-01


.. parsed-literal::

      68	 1.3223509e+00	 1.3253854e-01	 1.3727133e+00	 1.4792140e-01	[ 1.2896488e+00]	 2.0564771e-01


.. parsed-literal::

      69	 1.3315183e+00	 1.3125881e-01	 1.3820603e+00	 1.4684357e-01	[ 1.2926797e+00]	 2.0986724e-01


.. parsed-literal::

      70	 1.3387787e+00	 1.2981842e-01	 1.3896256e+00	 1.4580259e-01	  1.2908750e+00 	 2.0743322e-01


.. parsed-literal::

      71	 1.3453989e+00	 1.2923365e-01	 1.3962823e+00	 1.4541727e-01	[ 1.2945911e+00]	 2.1520734e-01


.. parsed-literal::

      72	 1.3529499e+00	 1.2860265e-01	 1.4039866e+00	 1.4511136e-01	[ 1.2962622e+00]	 2.1864462e-01
      73	 1.3605518e+00	 1.2835671e-01	 1.4118142e+00	 1.4526066e-01	  1.2937015e+00 	 1.9168711e-01


.. parsed-literal::

      74	 1.3668400e+00	 1.2785173e-01	 1.4184704e+00	 1.4612320e-01	  1.2755214e+00 	 1.7215919e-01
      75	 1.3749287e+00	 1.2776026e-01	 1.4263991e+00	 1.4602826e-01	  1.2855177e+00 	 1.8766737e-01


.. parsed-literal::

      76	 1.3794552e+00	 1.2754086e-01	 1.4309034e+00	 1.4598908e-01	  1.2900510e+00 	 2.1234965e-01


.. parsed-literal::

      77	 1.3858487e+00	 1.2697288e-01	 1.4375968e+00	 1.4596321e-01	  1.2892945e+00 	 2.1287894e-01


.. parsed-literal::

      78	 1.3898911e+00	 1.2640752e-01	 1.4418276e+00	 1.4610161e-01	  1.2891003e+00 	 3.2825923e-01


.. parsed-literal::

      79	 1.3946372e+00	 1.2594989e-01	 1.4465231e+00	 1.4607768e-01	  1.2953389e+00 	 2.0387936e-01


.. parsed-literal::

      80	 1.4000342e+00	 1.2531264e-01	 1.4519319e+00	 1.4611760e-01	  1.2947106e+00 	 2.1329474e-01


.. parsed-literal::

      81	 1.4054050e+00	 1.2491364e-01	 1.4573859e+00	 1.4602123e-01	[ 1.2973912e+00]	 2.0786881e-01


.. parsed-literal::

      82	 1.4098951e+00	 1.2445891e-01	 1.4618324e+00	 1.4606972e-01	  1.2964594e+00 	 2.1338749e-01
      83	 1.4132304e+00	 1.2453044e-01	 1.4651994e+00	 1.4618957e-01	  1.2959971e+00 	 1.9561958e-01


.. parsed-literal::

      84	 1.4177754e+00	 1.2445488e-01	 1.4698088e+00	 1.4613493e-01	  1.2950967e+00 	 2.0578647e-01


.. parsed-literal::

      85	 1.4223297e+00	 1.2459738e-01	 1.4745607e+00	 1.4678179e-01	  1.2952650e+00 	 2.0808077e-01


.. parsed-literal::

      86	 1.4270296e+00	 1.2435506e-01	 1.4792959e+00	 1.4669104e-01	  1.2955552e+00 	 2.0969486e-01
      87	 1.4302909e+00	 1.2414778e-01	 1.4825785e+00	 1.4665873e-01	[ 1.3028113e+00]	 1.9980478e-01


.. parsed-literal::

      88	 1.4338816e+00	 1.2390084e-01	 1.4862616e+00	 1.4671437e-01	[ 1.3110632e+00]	 1.9954133e-01


.. parsed-literal::

      89	 1.4374900e+00	 1.2367468e-01	 1.4899384e+00	 1.4633242e-01	[ 1.3190082e+00]	 2.0915151e-01
      90	 1.4418514e+00	 1.2338652e-01	 1.4943379e+00	 1.4608946e-01	[ 1.3199652e+00]	 1.7446423e-01


.. parsed-literal::

      91	 1.4454155e+00	 1.2332661e-01	 1.4979361e+00	 1.4587073e-01	  1.3150882e+00 	 2.0941567e-01


.. parsed-literal::

      92	 1.4484607e+00	 1.2302411e-01	 1.5009168e+00	 1.4569596e-01	  1.3095598e+00 	 2.1616101e-01


.. parsed-literal::

      93	 1.4508161e+00	 1.2296281e-01	 1.5032716e+00	 1.4568224e-01	  1.3085556e+00 	 2.2013593e-01


.. parsed-literal::

      94	 1.4539426e+00	 1.2270620e-01	 1.5064749e+00	 1.4565985e-01	  1.3029345e+00 	 2.1570492e-01


.. parsed-literal::

      95	 1.4567823e+00	 1.2269327e-01	 1.5093603e+00	 1.4537348e-01	  1.3057355e+00 	 2.2179627e-01


.. parsed-literal::

      96	 1.4592521e+00	 1.2258580e-01	 1.5118154e+00	 1.4519548e-01	  1.3087291e+00 	 2.1859145e-01


.. parsed-literal::

      97	 1.4630523e+00	 1.2235158e-01	 1.5156491e+00	 1.4505504e-01	  1.3078521e+00 	 2.1455145e-01
      98	 1.4652705e+00	 1.2224859e-01	 1.5178942e+00	 1.4495201e-01	  1.3091838e+00 	 1.8776631e-01


.. parsed-literal::

      99	 1.4673091e+00	 1.2221640e-01	 1.5198836e+00	 1.4504732e-01	  1.3065125e+00 	 1.9378090e-01


.. parsed-literal::

     100	 1.4697777e+00	 1.2227933e-01	 1.5223951e+00	 1.4517476e-01	  1.2989722e+00 	 2.0508766e-01
     101	 1.4717532e+00	 1.2232365e-01	 1.5244290e+00	 1.4512555e-01	  1.2938107e+00 	 1.8690896e-01


.. parsed-literal::

     102	 1.4751812e+00	 1.2237756e-01	 1.5280932e+00	 1.4510667e-01	  1.2765918e+00 	 2.0708537e-01


.. parsed-literal::

     103	 1.4776916e+00	 1.2248778e-01	 1.5306797e+00	 1.4476561e-01	  1.2700824e+00 	 2.0586491e-01


.. parsed-literal::

     104	 1.4795806e+00	 1.2236546e-01	 1.5324811e+00	 1.4478604e-01	  1.2756216e+00 	 2.0904756e-01


.. parsed-literal::

     105	 1.4811137e+00	 1.2220161e-01	 1.5339973e+00	 1.4484991e-01	  1.2766351e+00 	 2.0692205e-01


.. parsed-literal::

     106	 1.4829354e+00	 1.2219565e-01	 1.5358998e+00	 1.4495393e-01	  1.2733861e+00 	 2.1399713e-01
     107	 1.4849420e+00	 1.2212549e-01	 1.5379802e+00	 1.4531865e-01	  1.2692942e+00 	 1.9806147e-01


.. parsed-literal::

     108	 1.4863873e+00	 1.2224140e-01	 1.5394673e+00	 1.4536518e-01	  1.2686244e+00 	 2.1544099e-01


.. parsed-literal::

     109	 1.4880087e+00	 1.2235998e-01	 1.5411392e+00	 1.4546852e-01	  1.2686891e+00 	 2.0852065e-01
     110	 1.4898548e+00	 1.2248552e-01	 1.5430288e+00	 1.4555985e-01	  1.2750569e+00 	 1.8148112e-01


.. parsed-literal::

     111	 1.4924954e+00	 1.2242356e-01	 1.5457200e+00	 1.4570029e-01	  1.2803687e+00 	 1.8852353e-01
     112	 1.4951333e+00	 1.2232250e-01	 1.5483946e+00	 1.4583076e-01	  1.2877605e+00 	 1.8091798e-01


.. parsed-literal::

     113	 1.4973989e+00	 1.2204376e-01	 1.5506775e+00	 1.4584676e-01	  1.2867337e+00 	 2.0897245e-01


.. parsed-literal::

     114	 1.4993860e+00	 1.2193106e-01	 1.5526465e+00	 1.4577107e-01	  1.2841910e+00 	 2.1171570e-01


.. parsed-literal::

     115	 1.5013790e+00	 1.2195554e-01	 1.5546904e+00	 1.4581245e-01	  1.2747318e+00 	 2.0405912e-01
     116	 1.5032124e+00	 1.2204854e-01	 1.5566520e+00	 1.4567741e-01	  1.2643852e+00 	 2.0280457e-01


.. parsed-literal::

     117	 1.5050428e+00	 1.2211792e-01	 1.5585234e+00	 1.4564170e-01	  1.2592639e+00 	 2.1697569e-01


.. parsed-literal::

     118	 1.5065621e+00	 1.2211970e-01	 1.5600717e+00	 1.4556125e-01	  1.2571791e+00 	 2.1872067e-01


.. parsed-literal::

     119	 1.5075818e+00	 1.2202305e-01	 1.5610847e+00	 1.4522136e-01	  1.2589033e+00 	 2.1797895e-01


.. parsed-literal::

     120	 1.5084467e+00	 1.2191047e-01	 1.5619410e+00	 1.4510029e-01	  1.2595289e+00 	 2.1596360e-01


.. parsed-literal::

     121	 1.5104265e+00	 1.2159328e-01	 1.5639374e+00	 1.4463792e-01	  1.2560498e+00 	 2.1021438e-01
     122	 1.5119939e+00	 1.2138941e-01	 1.5655602e+00	 1.4426849e-01	  1.2504930e+00 	 2.0326972e-01


.. parsed-literal::

     123	 1.5142726e+00	 1.2100775e-01	 1.5679600e+00	 1.4339198e-01	  1.2362659e+00 	 1.9575953e-01


.. parsed-literal::

     124	 1.5162527e+00	 1.2089034e-01	 1.5700217e+00	 1.4306203e-01	  1.2311344e+00 	 2.4716473e-01


.. parsed-literal::

     125	 1.5179556e+00	 1.2081010e-01	 1.5717851e+00	 1.4285704e-01	  1.2266896e+00 	 2.1701741e-01


.. parsed-literal::

     126	 1.5194559e+00	 1.2075272e-01	 1.5733985e+00	 1.4236909e-01	  1.2225506e+00 	 2.0267677e-01


.. parsed-literal::

     127	 1.5211740e+00	 1.2057740e-01	 1.5751162e+00	 1.4225361e-01	  1.2167599e+00 	 2.1463752e-01


.. parsed-literal::

     128	 1.5224894e+00	 1.2042436e-01	 1.5764240e+00	 1.4210160e-01	  1.2137167e+00 	 2.1128869e-01


.. parsed-literal::

     129	 1.5244597e+00	 1.2023556e-01	 1.5784043e+00	 1.4186665e-01	  1.2098923e+00 	 2.0951390e-01


.. parsed-literal::

     130	 1.5252257e+00	 1.2014336e-01	 1.5791960e+00	 1.4182549e-01	  1.2082667e+00 	 3.2326388e-01
     131	 1.5265487e+00	 1.2004436e-01	 1.5805311e+00	 1.4160882e-01	  1.2072455e+00 	 1.8422127e-01


.. parsed-literal::

     132	 1.5277382e+00	 1.1997916e-01	 1.5817424e+00	 1.4145882e-01	  1.2081207e+00 	 1.9748187e-01


.. parsed-literal::

     133	 1.5292736e+00	 1.1986436e-01	 1.5833184e+00	 1.4116586e-01	  1.2052148e+00 	 2.1425390e-01


.. parsed-literal::

     134	 1.5307969e+00	 1.1979688e-01	 1.5848916e+00	 1.4099102e-01	  1.2028293e+00 	 2.0923615e-01
     135	 1.5322742e+00	 1.1969043e-01	 1.5863963e+00	 1.4072142e-01	  1.1955915e+00 	 2.0711875e-01


.. parsed-literal::

     136	 1.5340875e+00	 1.1956431e-01	 1.5882493e+00	 1.4034296e-01	  1.1842316e+00 	 2.0494699e-01


.. parsed-literal::

     137	 1.5345928e+00	 1.1946989e-01	 1.5888704e+00	 1.4005704e-01	  1.1721991e+00 	 2.1399856e-01


.. parsed-literal::

     138	 1.5360145e+00	 1.1944680e-01	 1.5902083e+00	 1.3996418e-01	  1.1756264e+00 	 2.0568967e-01
     139	 1.5366550e+00	 1.1941976e-01	 1.5908410e+00	 1.3989085e-01	  1.1769282e+00 	 1.8423867e-01


.. parsed-literal::

     140	 1.5377784e+00	 1.1935706e-01	 1.5919745e+00	 1.3978241e-01	  1.1767662e+00 	 2.0225549e-01


.. parsed-literal::

     141	 1.5394417e+00	 1.1928108e-01	 1.5936687e+00	 1.3969402e-01	  1.1723334e+00 	 2.1371865e-01


.. parsed-literal::

     142	 1.5402950e+00	 1.1919770e-01	 1.5945499e+00	 1.3969111e-01	  1.1679285e+00 	 3.2433057e-01


.. parsed-literal::

     143	 1.5416728e+00	 1.1916289e-01	 1.5959490e+00	 1.3971655e-01	  1.1601296e+00 	 2.0723081e-01


.. parsed-literal::

     144	 1.5425945e+00	 1.1913071e-01	 1.5968977e+00	 1.3973848e-01	  1.1531553e+00 	 2.2019506e-01


.. parsed-literal::

     145	 1.5433481e+00	 1.1898583e-01	 1.5976962e+00	 1.3999305e-01	  1.1437563e+00 	 2.2131968e-01


.. parsed-literal::

     146	 1.5443156e+00	 1.1893427e-01	 1.5986572e+00	 1.3990501e-01	  1.1418468e+00 	 2.1065116e-01


.. parsed-literal::

     147	 1.5452754e+00	 1.1882606e-01	 1.5996400e+00	 1.3982126e-01	  1.1371477e+00 	 2.1259999e-01


.. parsed-literal::

     148	 1.5459255e+00	 1.1873296e-01	 1.6003162e+00	 1.3977638e-01	  1.1330584e+00 	 2.0615482e-01


.. parsed-literal::

     149	 1.5474821e+00	 1.1851819e-01	 1.6019366e+00	 1.3969265e-01	  1.1233841e+00 	 2.0276594e-01


.. parsed-literal::

     150	 1.5483434e+00	 1.1839547e-01	 1.6028580e+00	 1.3957888e-01	  1.1214304e+00 	 3.1230021e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.18 s, total: 2min 7s
    Wall time: 32.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7ff8743b5150>



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
    CPU times: user 1.83 s, sys: 44 ms, total: 1.87 s
    Wall time: 627 ms


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

