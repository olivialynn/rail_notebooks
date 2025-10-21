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

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4755064e-01	 3.2158697e-01	-3.3781568e-01	 3.1642674e-01	[-3.2745381e-01]	 4.8419571e-01


.. parsed-literal::

       2	-2.7668714e-01	 3.1125775e-01	-2.5298146e-01	 3.0578261e-01	[-2.3637915e-01]	 2.3707199e-01


.. parsed-literal::

       3	-2.3183478e-01	 2.9038169e-01	-1.9010346e-01	 2.8500929e-01	[-1.6828041e-01]	 2.8740454e-01
       4	-2.0037330e-01	 2.6728335e-01	-1.5915341e-01	 2.6623163e-01	[-1.4893353e-01]	 1.7965722e-01


.. parsed-literal::

       5	-1.0978771e-01	 2.5855207e-01	-7.4909963e-02	 2.5596155e-01	[-6.2895959e-02]	 2.1452236e-01
       6	-7.5916815e-02	 2.5298301e-01	-4.5333911e-02	 2.5009427e-01	[-3.4977889e-02]	 1.8615127e-01


.. parsed-literal::

       7	-5.7287674e-02	 2.5005414e-01	-3.3147819e-02	 2.4724700e-01	[-2.2225070e-02]	 1.9710994e-01
       8	-4.4882118e-02	 2.4796882e-01	-2.4504345e-02	 2.4469274e-01	[-1.1588902e-02]	 1.8263960e-01


.. parsed-literal::

       9	-3.1334837e-02	 2.4545332e-01	-1.3825104e-02	 2.4160138e-01	[ 1.4384627e-03]	 1.9788933e-01


.. parsed-literal::

      10	-1.8073882e-02	 2.4280814e-01	-2.4821246e-03	 2.3975091e-01	[ 1.0071855e-02]	 2.0957255e-01


.. parsed-literal::

      11	-1.3525494e-02	 2.4232961e-01	 3.4508299e-04	 2.3936622e-01	[ 1.1797597e-02]	 2.1314359e-01


.. parsed-literal::

      12	-9.4286183e-03	 2.4151111e-01	 4.2593653e-03	 2.3883663e-01	[ 1.4766431e-02]	 2.2286534e-01


.. parsed-literal::

      13	-5.9073212e-03	 2.4077602e-01	 7.8247190e-03	 2.3859954e-01	[ 1.6291064e-02]	 2.1453142e-01


.. parsed-literal::

      14	 3.8157768e-03	 2.3858071e-01	 1.9262710e-02	 2.3646659e-01	[ 2.7653780e-02]	 2.1998215e-01
      15	 2.0303253e-01	 2.2181886e-01	 2.2898385e-01	 2.2066940e-01	[ 2.3221767e-01]	 1.9950795e-01


.. parsed-literal::

      16	 2.2924965e-01	 2.2364536e-01	 2.5512568e-01	 2.2249260e-01	[ 2.4668370e-01]	 3.1909657e-01


.. parsed-literal::

      17	 3.0500499e-01	 2.1575245e-01	 3.3405069e-01	 2.1706839e-01	[ 3.1855294e-01]	 2.1017933e-01


.. parsed-literal::

      18	 3.7805668e-01	 2.0916487e-01	 4.1029395e-01	 2.1600955e-01	[ 3.6761890e-01]	 3.2831526e-01


.. parsed-literal::

      19	 4.2414865e-01	 2.0427199e-01	 4.5758976e-01	 2.1016027e-01	[ 4.1784586e-01]	 2.1224546e-01


.. parsed-literal::

      20	 4.5621187e-01	 2.0316240e-01	 4.8993357e-01	 2.1179860e-01	[ 4.3922924e-01]	 2.1544051e-01


.. parsed-literal::

      21	 4.8638984e-01	 2.0081884e-01	 5.2045094e-01	 2.1052319e-01	[ 4.6244000e-01]	 2.1477199e-01
      22	 5.6221547e-01	 1.9783926e-01	 5.9790163e-01	 2.0674575e-01	[ 5.2818034e-01]	 1.9861531e-01


.. parsed-literal::

      23	 6.3206024e-01	 1.9604488e-01	 6.7062454e-01	 2.0460433e-01	[ 5.9444587e-01]	 2.0186830e-01


.. parsed-literal::

      24	 6.6474740e-01	 1.9507955e-01	 7.0444311e-01	 2.0166178e-01	[ 6.1357665e-01]	 2.0765591e-01


.. parsed-literal::

      25	 7.0247235e-01	 1.9024466e-01	 7.4080083e-01	 1.9845595e-01	[ 6.5801479e-01]	 2.1089387e-01


.. parsed-literal::

      26	 7.3221416e-01	 1.8997336e-01	 7.6949195e-01	 1.9718664e-01	[ 6.9421924e-01]	 2.1275306e-01
      27	 7.4776506e-01	 1.9601037e-01	 7.8409746e-01	 2.0020098e-01	[ 7.1230634e-01]	 1.7995548e-01


.. parsed-literal::

      28	 7.7364521e-01	 1.8961801e-01	 8.1160375e-01	 1.9491971e-01	[ 7.2531593e-01]	 1.7593980e-01


.. parsed-literal::

      29	 7.9726807e-01	 1.8833555e-01	 8.3567470e-01	 1.9460958e-01	[ 7.4068968e-01]	 2.0762324e-01


.. parsed-literal::

      30	 8.2239161e-01	 1.8749815e-01	 8.6098924e-01	 1.9330824e-01	[ 7.6476325e-01]	 2.1897531e-01


.. parsed-literal::

      31	 8.5438246e-01	 1.8686242e-01	 8.9372277e-01	 1.9251349e-01	[ 8.0760220e-01]	 2.0449805e-01


.. parsed-literal::

      32	 8.7916510e-01	 1.8453893e-01	 9.1826060e-01	 1.8665206e-01	[ 8.4619904e-01]	 2.1146941e-01


.. parsed-literal::

      33	 8.9317736e-01	 1.8160516e-01	 9.3272563e-01	 1.8542062e-01	[ 8.5366405e-01]	 2.0249581e-01


.. parsed-literal::

      34	 9.0547167e-01	 1.8028119e-01	 9.4547432e-01	 1.8492268e-01	[ 8.6795805e-01]	 2.1333981e-01


.. parsed-literal::

      35	 9.1771186e-01	 1.7820198e-01	 9.5985591e-01	 1.8255094e-01	[ 8.7886173e-01]	 2.1454406e-01
      36	 9.3635532e-01	 1.7882402e-01	 9.7791091e-01	 1.8386351e-01	[ 9.0767135e-01]	 1.9650483e-01


.. parsed-literal::

      37	 9.4840110e-01	 1.7730822e-01	 9.8985027e-01	 1.8369072e-01	[ 9.1590339e-01]	 2.0284081e-01


.. parsed-literal::

      38	 9.6598383e-01	 1.7547124e-01	 1.0078684e+00	 1.8190315e-01	[ 9.2803413e-01]	 2.1022749e-01
      39	 9.8606875e-01	 1.7410376e-01	 1.0289994e+00	 1.7921078e-01	[ 9.4462397e-01]	 1.9636703e-01


.. parsed-literal::

      40	 9.9880235e-01	 1.7189711e-01	 1.0425032e+00	 1.7798986e-01	[ 9.5477880e-01]	 2.1288538e-01
      41	 1.0116125e+00	 1.7052218e-01	 1.0557750e+00	 1.7669839e-01	[ 9.6722763e-01]	 1.8919635e-01


.. parsed-literal::

      42	 1.0302651e+00	 1.7075982e-01	 1.0753286e+00	 1.7508327e-01	[ 9.8208130e-01]	 2.0678949e-01


.. parsed-literal::

      43	 1.0388569e+00	 1.7074666e-01	 1.0843659e+00	 1.7586069e-01	[ 9.9323194e-01]	 2.0093703e-01


.. parsed-literal::

      44	 1.0474084e+00	 1.7046567e-01	 1.0922597e+00	 1.7603609e-01	[ 1.0031060e+00]	 2.1700358e-01


.. parsed-literal::

      45	 1.0551083e+00	 1.7017009e-01	 1.1001796e+00	 1.7603454e-01	[ 1.0130620e+00]	 2.2085571e-01
      46	 1.0618956e+00	 1.6983366e-01	 1.1074008e+00	 1.7627677e-01	[ 1.0215676e+00]	 1.7526889e-01


.. parsed-literal::

      47	 1.0779931e+00	 1.6810949e-01	 1.1247694e+00	 1.7546272e-01	[ 1.0356843e+00]	 2.1664691e-01


.. parsed-literal::

      48	 1.0860809e+00	 1.6696308e-01	 1.1331573e+00	 1.7396952e-01	[ 1.0447765e+00]	 3.4037161e-01


.. parsed-literal::

      49	 1.0941589e+00	 1.6605480e-01	 1.1411815e+00	 1.7316049e-01	[ 1.0482052e+00]	 2.0823288e-01


.. parsed-literal::

      50	 1.1060829e+00	 1.6407572e-01	 1.1530504e+00	 1.7150827e-01	[ 1.0532052e+00]	 2.1139288e-01


.. parsed-literal::

      51	 1.1155058e+00	 1.6284025e-01	 1.1626268e+00	 1.6951768e-01	  1.0530887e+00 	 2.1019101e-01


.. parsed-literal::

      52	 1.1260225e+00	 1.6152579e-01	 1.1730118e+00	 1.6804318e-01	[ 1.0648073e+00]	 2.1007466e-01
      53	 1.1345367e+00	 1.6028422e-01	 1.1817556e+00	 1.6662581e-01	[ 1.0754404e+00]	 1.8139839e-01


.. parsed-literal::

      54	 1.1439642e+00	 1.5898413e-01	 1.1914252e+00	 1.6524336e-01	[ 1.0916721e+00]	 1.7214298e-01
      55	 1.1540321e+00	 1.5675275e-01	 1.2019579e+00	 1.6399185e-01	[ 1.1047279e+00]	 1.9082141e-01


.. parsed-literal::

      56	 1.1624763e+00	 1.5566060e-01	 1.2103367e+00	 1.6322226e-01	[ 1.1139788e+00]	 2.0859241e-01


.. parsed-literal::

      57	 1.1719650e+00	 1.5392536e-01	 1.2199631e+00	 1.6237276e-01	[ 1.1191346e+00]	 2.2188735e-01


.. parsed-literal::

      58	 1.1808074e+00	 1.5231095e-01	 1.2290092e+00	 1.6135013e-01	[ 1.1286132e+00]	 2.1551418e-01
      59	 1.1903329e+00	 1.5077566e-01	 1.2387022e+00	 1.6025690e-01	[ 1.1354054e+00]	 1.8341660e-01


.. parsed-literal::

      60	 1.2003305e+00	 1.4880875e-01	 1.2490566e+00	 1.5873529e-01	[ 1.1451422e+00]	 2.0903230e-01


.. parsed-literal::

      61	 1.2057770e+00	 1.4766792e-01	 1.2549659e+00	 1.5740683e-01	[ 1.1563857e+00]	 2.0524263e-01


.. parsed-literal::

      62	 1.2148603e+00	 1.4690897e-01	 1.2639329e+00	 1.5674656e-01	[ 1.1627660e+00]	 2.1341491e-01


.. parsed-literal::

      63	 1.2203379e+00	 1.4661612e-01	 1.2693237e+00	 1.5670012e-01	[ 1.1660089e+00]	 2.2091889e-01
      64	 1.2256946e+00	 1.4581226e-01	 1.2750131e+00	 1.5566142e-01	[ 1.1683581e+00]	 1.9341707e-01


.. parsed-literal::

      65	 1.2324929e+00	 1.4546460e-01	 1.2818729e+00	 1.5532814e-01	[ 1.1719118e+00]	 2.1747494e-01
      66	 1.2404414e+00	 1.4477174e-01	 1.2900975e+00	 1.5457845e-01	[ 1.1740040e+00]	 1.8058944e-01


.. parsed-literal::

      67	 1.2478393e+00	 1.4402449e-01	 1.2980916e+00	 1.5361920e-01	[ 1.1750482e+00]	 1.8609405e-01


.. parsed-literal::

      68	 1.2543497e+00	 1.4366990e-01	 1.3045902e+00	 1.5361156e-01	[ 1.1773807e+00]	 2.0689774e-01


.. parsed-literal::

      69	 1.2588915e+00	 1.4329340e-01	 1.3091065e+00	 1.5345677e-01	[ 1.1791748e+00]	 2.2029209e-01


.. parsed-literal::

      70	 1.2657399e+00	 1.4253356e-01	 1.3158887e+00	 1.5320137e-01	[ 1.1877168e+00]	 2.0789862e-01


.. parsed-literal::

      71	 1.2710966e+00	 1.4219582e-01	 1.3215013e+00	 1.5269725e-01	[ 1.1928627e+00]	 2.1556497e-01
      72	 1.2768724e+00	 1.4184754e-01	 1.3272322e+00	 1.5254646e-01	[ 1.1997309e+00]	 1.8124104e-01


.. parsed-literal::

      73	 1.2831592e+00	 1.4132176e-01	 1.3338214e+00	 1.5192258e-01	[ 1.2035170e+00]	 2.0182276e-01


.. parsed-literal::

      74	 1.2891838e+00	 1.4103974e-01	 1.3400525e+00	 1.5152824e-01	[ 1.2109808e+00]	 2.0709682e-01


.. parsed-literal::

      75	 1.2967299e+00	 1.3994605e-01	 1.3482589e+00	 1.5040857e-01	  1.2021622e+00 	 2.1495223e-01


.. parsed-literal::

      76	 1.3036203e+00	 1.3976288e-01	 1.3549436e+00	 1.5015750e-01	[ 1.2153560e+00]	 2.1246624e-01


.. parsed-literal::

      77	 1.3085073e+00	 1.3944049e-01	 1.3595215e+00	 1.5021047e-01	[ 1.2216905e+00]	 2.1856976e-01


.. parsed-literal::

      78	 1.3172951e+00	 1.3848055e-01	 1.3683533e+00	 1.5001420e-01	  1.2199614e+00 	 2.2181773e-01


.. parsed-literal::

      79	 1.3228727e+00	 1.3771107e-01	 1.3741069e+00	 1.4975634e-01	  1.2214813e+00 	 2.1084428e-01
      80	 1.3288198e+00	 1.3719840e-01	 1.3800164e+00	 1.4956881e-01	[ 1.2237703e+00]	 1.9972587e-01


.. parsed-literal::

      81	 1.3340085e+00	 1.3684524e-01	 1.3853479e+00	 1.4936373e-01	  1.2219224e+00 	 2.0936155e-01
      82	 1.3384146e+00	 1.3672254e-01	 1.3897899e+00	 1.4921335e-01	[ 1.2277556e+00]	 1.7804217e-01


.. parsed-literal::

      83	 1.3448328e+00	 1.3651575e-01	 1.3963770e+00	 1.4909758e-01	[ 1.2299147e+00]	 2.0038772e-01
      84	 1.3503281e+00	 1.3642906e-01	 1.4019468e+00	 1.4884623e-01	[ 1.2325190e+00]	 1.9102073e-01


.. parsed-literal::

      85	 1.3554044e+00	 1.3616915e-01	 1.4069924e+00	 1.4862405e-01	[ 1.2387037e+00]	 2.1571970e-01


.. parsed-literal::

      86	 1.3592705e+00	 1.3606670e-01	 1.4108333e+00	 1.4845472e-01	  1.2385830e+00 	 2.0975375e-01


.. parsed-literal::

      87	 1.3654003e+00	 1.3621589e-01	 1.4171547e+00	 1.4796415e-01	  1.2332503e+00 	 2.0830512e-01


.. parsed-literal::

      88	 1.3672878e+00	 1.3633318e-01	 1.4195103e+00	 1.4769407e-01	  1.2171864e+00 	 2.1244073e-01


.. parsed-literal::

      89	 1.3733387e+00	 1.3637300e-01	 1.4253818e+00	 1.4766805e-01	  1.2221724e+00 	 2.0494056e-01


.. parsed-literal::

      90	 1.3751117e+00	 1.3632131e-01	 1.4271713e+00	 1.4767789e-01	  1.2228061e+00 	 2.0376062e-01


.. parsed-literal::

      91	 1.3781025e+00	 1.3620136e-01	 1.4302771e+00	 1.4776621e-01	  1.2194670e+00 	 2.1928859e-01


.. parsed-literal::

      92	 1.3822822e+00	 1.3561539e-01	 1.4346543e+00	 1.4744749e-01	  1.2153763e+00 	 2.2212768e-01


.. parsed-literal::

      93	 1.3864119e+00	 1.3536573e-01	 1.4391034e+00	 1.4756338e-01	  1.1999878e+00 	 2.1408224e-01


.. parsed-literal::

      94	 1.3899863e+00	 1.3481328e-01	 1.4426682e+00	 1.4704928e-01	  1.1999217e+00 	 2.0866585e-01
      95	 1.3939540e+00	 1.3404099e-01	 1.4467438e+00	 1.4579695e-01	  1.1959874e+00 	 1.9285321e-01


.. parsed-literal::

      96	 1.3969653e+00	 1.3385502e-01	 1.4498149e+00	 1.4501607e-01	  1.1963305e+00 	 2.0982575e-01


.. parsed-literal::

      97	 1.4003584e+00	 1.3353090e-01	 1.4532686e+00	 1.4473284e-01	  1.1932985e+00 	 2.1877575e-01


.. parsed-literal::

      98	 1.4048499e+00	 1.3290713e-01	 1.4579412e+00	 1.4360377e-01	  1.1823523e+00 	 2.3307514e-01
      99	 1.4082205e+00	 1.3254711e-01	 1.4613073e+00	 1.4351926e-01	  1.1781527e+00 	 1.9138312e-01


.. parsed-literal::

     100	 1.4110973e+00	 1.3218012e-01	 1.4641132e+00	 1.4321275e-01	  1.1785122e+00 	 1.6854262e-01


.. parsed-literal::

     101	 1.4158961e+00	 1.3162808e-01	 1.4689778e+00	 1.4229610e-01	  1.1704965e+00 	 2.1101403e-01


.. parsed-literal::

     102	 1.4197173e+00	 1.3150126e-01	 1.4727963e+00	 1.4174452e-01	  1.1723190e+00 	 2.0515871e-01


.. parsed-literal::

     103	 1.4238274e+00	 1.3135790e-01	 1.4769825e+00	 1.4069358e-01	  1.1681402e+00 	 2.1318793e-01


.. parsed-literal::

     104	 1.4266155e+00	 1.3165244e-01	 1.4798380e+00	 1.4049987e-01	  1.1665441e+00 	 2.1389222e-01
     105	 1.4285017e+00	 1.3153741e-01	 1.4816933e+00	 1.4055324e-01	  1.1696021e+00 	 1.9071245e-01


.. parsed-literal::

     106	 1.4313305e+00	 1.3127656e-01	 1.4845940e+00	 1.4039124e-01	  1.1712586e+00 	 1.7983890e-01


.. parsed-literal::

     107	 1.4348686e+00	 1.3076397e-01	 1.4882581e+00	 1.4003457e-01	  1.1730495e+00 	 2.0867777e-01
     108	 1.4372487e+00	 1.3046584e-01	 1.4909408e+00	 1.3940442e-01	  1.1675589e+00 	 2.0007586e-01


.. parsed-literal::

     109	 1.4424465e+00	 1.2982058e-01	 1.4959970e+00	 1.3900381e-01	  1.1743164e+00 	 1.7562246e-01


.. parsed-literal::

     110	 1.4447372e+00	 1.2958754e-01	 1.4982236e+00	 1.3875528e-01	  1.1744905e+00 	 2.1063423e-01


.. parsed-literal::

     111	 1.4481802e+00	 1.2922355e-01	 1.5017442e+00	 1.3838496e-01	  1.1687358e+00 	 2.1711469e-01


.. parsed-literal::

     112	 1.4514410e+00	 1.2870589e-01	 1.5052230e+00	 1.3822022e-01	  1.1498953e+00 	 2.1067786e-01
     113	 1.4554930e+00	 1.2855187e-01	 1.5093654e+00	 1.3795253e-01	  1.1495383e+00 	 1.8676114e-01


.. parsed-literal::

     114	 1.4578492e+00	 1.2845979e-01	 1.5117527e+00	 1.3806850e-01	  1.1468967e+00 	 1.9982815e-01


.. parsed-literal::

     115	 1.4603226e+00	 1.2817979e-01	 1.5143006e+00	 1.3795932e-01	  1.1416382e+00 	 2.0574999e-01


.. parsed-literal::

     116	 1.4620211e+00	 1.2824695e-01	 1.5161196e+00	 1.3832269e-01	  1.1348836e+00 	 2.1015692e-01
     117	 1.4649462e+00	 1.2775375e-01	 1.5189565e+00	 1.3776885e-01	  1.1352078e+00 	 1.8934345e-01


.. parsed-literal::

     118	 1.4675052e+00	 1.2726630e-01	 1.5214694e+00	 1.3711079e-01	  1.1350501e+00 	 2.1974707e-01
     119	 1.4695816e+00	 1.2694654e-01	 1.5235475e+00	 1.3669758e-01	  1.1339240e+00 	 2.0722556e-01


.. parsed-literal::

     120	 1.4722902e+00	 1.2642568e-01	 1.5264615e+00	 1.3561458e-01	  1.1210978e+00 	 2.1439719e-01


.. parsed-literal::

     121	 1.4758972e+00	 1.2620723e-01	 1.5300617e+00	 1.3584618e-01	  1.1175163e+00 	 2.1658635e-01
     122	 1.4775706e+00	 1.2623271e-01	 1.5317734e+00	 1.3601379e-01	  1.1155138e+00 	 2.0313358e-01


.. parsed-literal::

     123	 1.4796049e+00	 1.2612966e-01	 1.5339110e+00	 1.3612952e-01	  1.1069114e+00 	 2.1709228e-01


.. parsed-literal::

     124	 1.4809469e+00	 1.2614174e-01	 1.5353704e+00	 1.3601118e-01	  1.0971356e+00 	 2.1037364e-01


.. parsed-literal::

     125	 1.4830388e+00	 1.2590400e-01	 1.5373687e+00	 1.3584667e-01	  1.0992239e+00 	 2.1191216e-01


.. parsed-literal::

     126	 1.4852472e+00	 1.2558842e-01	 1.5395443e+00	 1.3554509e-01	  1.0947063e+00 	 2.1009779e-01


.. parsed-literal::

     127	 1.4872865e+00	 1.2547133e-01	 1.5415579e+00	 1.3525172e-01	  1.0911359e+00 	 2.0881534e-01
     128	 1.4895852e+00	 1.2527871e-01	 1.5440201e+00	 1.3478674e-01	  1.0670812e+00 	 1.8530059e-01


.. parsed-literal::

     129	 1.4934515e+00	 1.2548341e-01	 1.5477885e+00	 1.3444573e-01	  1.0739156e+00 	 2.1038747e-01
     130	 1.4950855e+00	 1.2554672e-01	 1.5494266e+00	 1.3430404e-01	  1.0725845e+00 	 1.9954562e-01


.. parsed-literal::

     131	 1.4972667e+00	 1.2570088e-01	 1.5517109e+00	 1.3400341e-01	  1.0601969e+00 	 2.1227455e-01


.. parsed-literal::

     132	 1.4988425e+00	 1.2552965e-01	 1.5534214e+00	 1.3319326e-01	  1.0418281e+00 	 2.0904779e-01


.. parsed-literal::

     133	 1.5010878e+00	 1.2554351e-01	 1.5556537e+00	 1.3326907e-01	  1.0354964e+00 	 2.0104289e-01


.. parsed-literal::

     134	 1.5027015e+00	 1.2550807e-01	 1.5572741e+00	 1.3321759e-01	  1.0302085e+00 	 2.1073866e-01


.. parsed-literal::

     135	 1.5038751e+00	 1.2542092e-01	 1.5584420e+00	 1.3309835e-01	  1.0282207e+00 	 2.0222473e-01


.. parsed-literal::

     136	 1.5048106e+00	 1.2537768e-01	 1.5595413e+00	 1.3308585e-01	  1.0218931e+00 	 2.1694326e-01


.. parsed-literal::

     137	 1.5076221e+00	 1.2529401e-01	 1.5622169e+00	 1.3274620e-01	  1.0212811e+00 	 2.1357727e-01
     138	 1.5087755e+00	 1.2523321e-01	 1.5633531e+00	 1.3267532e-01	  1.0204877e+00 	 1.9839931e-01


.. parsed-literal::

     139	 1.5104128e+00	 1.2519315e-01	 1.5650504e+00	 1.3255794e-01	  1.0110599e+00 	 2.0914960e-01
     140	 1.5130055e+00	 1.2500687e-01	 1.5677518e+00	 1.3246000e-01	  9.9295683e-01 	 1.7955685e-01


.. parsed-literal::

     141	 1.5145541e+00	 1.2507975e-01	 1.5695282e+00	 1.3213769e-01	  9.5495490e-01 	 2.1507096e-01
     142	 1.5165262e+00	 1.2492610e-01	 1.5714070e+00	 1.3220010e-01	  9.6623948e-01 	 1.8384242e-01


.. parsed-literal::

     143	 1.5174891e+00	 1.2485621e-01	 1.5723209e+00	 1.3219561e-01	  9.7316502e-01 	 2.0836639e-01


.. parsed-literal::

     144	 1.5195634e+00	 1.2464083e-01	 1.5743921e+00	 1.3216312e-01	  9.7879053e-01 	 2.0858526e-01


.. parsed-literal::

     145	 1.5209343e+00	 1.2443966e-01	 1.5757900e+00	 1.3200743e-01	  9.8343983e-01 	 3.2898092e-01


.. parsed-literal::

     146	 1.5225754e+00	 1.2426066e-01	 1.5774883e+00	 1.3200112e-01	  9.7668037e-01 	 2.1279526e-01


.. parsed-literal::

     147	 1.5240778e+00	 1.2408715e-01	 1.5790672e+00	 1.3192874e-01	  9.6180527e-01 	 2.0252466e-01
     148	 1.5257599e+00	 1.2388598e-01	 1.5808330e+00	 1.3201033e-01	  9.3825580e-01 	 1.9093847e-01


.. parsed-literal::

     149	 1.5272213e+00	 1.2362575e-01	 1.5824783e+00	 1.3176518e-01	  9.0359269e-01 	 2.0119357e-01


.. parsed-literal::

     150	 1.5290056e+00	 1.2357311e-01	 1.5842022e+00	 1.3202924e-01	  9.0042193e-01 	 2.1270919e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.03 s, total: 2min 7s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fcea038e380>



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
    CPU times: user 1.88 s, sys: 44.9 ms, total: 1.93 s
    Wall time: 638 ms


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

