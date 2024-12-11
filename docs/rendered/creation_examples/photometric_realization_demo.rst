Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f1ee2476ad0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.890625</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.708771</td>
          <td>0.165625</td>
          <td>25.978472</td>
          <td>0.077362</td>
          <td>25.386870</td>
          <td>0.074804</td>
          <td>25.023556</td>
          <td>0.103558</td>
          <td>25.208277</td>
          <td>0.264664</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.381108</td>
          <td>0.618406</td>
          <td>27.344198</td>
          <td>0.249953</td>
          <td>27.479197</td>
          <td>0.431428</td>
          <td>26.514386</td>
          <td>0.360729</td>
          <td>25.929632</td>
          <td>0.466118</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.399965</td>
          <td>0.348671</td>
          <td>26.162116</td>
          <td>0.103280</td>
          <td>24.817804</td>
          <td>0.027705</td>
          <td>23.874636</td>
          <td>0.019801</td>
          <td>23.138036</td>
          <td>0.019750</td>
          <td>22.800563</td>
          <td>0.032700</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.334107</td>
          <td>0.693634</td>
          <td>27.294183</td>
          <td>0.269948</td>
          <td>27.310771</td>
          <td>0.243170</td>
          <td>26.697255</td>
          <td>0.231420</td>
          <td>26.371507</td>
          <td>0.322234</td>
          <td>25.383969</td>
          <td>0.305116</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.195229</td>
          <td>0.296270</td>
          <td>25.767981</td>
          <td>0.073041</td>
          <td>25.429027</td>
          <td>0.047530</td>
          <td>24.773594</td>
          <td>0.043420</td>
          <td>24.352073</td>
          <td>0.057255</td>
          <td>23.693783</td>
          <td>0.072154</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>2.147172</td>
          <td>26.419146</td>
          <td>0.353964</td>
          <td>26.107545</td>
          <td>0.098465</td>
          <td>26.003437</td>
          <td>0.079086</td>
          <td>26.200554</td>
          <td>0.152178</td>
          <td>25.976107</td>
          <td>0.233683</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.309317</td>
          <td>0.324562</td>
          <td>26.828895</td>
          <td>0.183404</td>
          <td>26.895757</td>
          <td>0.171743</td>
          <td>26.226860</td>
          <td>0.155648</td>
          <td>26.256103</td>
          <td>0.293770</td>
          <td>25.325477</td>
          <td>0.291086</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.847067</td>
          <td>0.417890</td>
          <td>26.801128</td>
          <td>0.158428</td>
          <td>26.444337</td>
          <td>0.187276</td>
          <td>26.054578</td>
          <td>0.249309</td>
          <td>25.616362</td>
          <td>0.366767</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.092102</td>
          <td>0.228638</td>
          <td>26.508182</td>
          <td>0.123075</td>
          <td>25.934227</td>
          <td>0.120916</td>
          <td>25.385130</td>
          <td>0.141763</td>
          <td>24.699962</td>
          <td>0.173203</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.732543</td>
          <td>0.450441</td>
          <td>26.253811</td>
          <td>0.111883</td>
          <td>26.214032</td>
          <td>0.095197</td>
          <td>25.657907</td>
          <td>0.094983</td>
          <td>25.170478</td>
          <td>0.117721</td>
          <td>24.433729</td>
          <td>0.137889</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.890625</td>
          <td>28.094275</td>
          <td>1.205691</td>
          <td>26.576514</td>
          <td>0.169982</td>
          <td>25.959716</td>
          <td>0.089500</td>
          <td>25.421024</td>
          <td>0.091302</td>
          <td>25.071065</td>
          <td>0.126670</td>
          <td>24.972009</td>
          <td>0.254818</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.372727</td>
          <td>0.778299</td>
          <td>29.589538</td>
          <td>1.424214</td>
          <td>27.876793</td>
          <td>0.440670</td>
          <td>26.891880</td>
          <td>0.316959</td>
          <td>25.950141</td>
          <td>0.266062</td>
          <td>25.328688</td>
          <td>0.339707</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.052243</td>
          <td>0.298682</td>
          <td>25.883265</td>
          <td>0.095220</td>
          <td>24.817388</td>
          <td>0.033316</td>
          <td>23.888323</td>
          <td>0.024182</td>
          <td>23.134651</td>
          <td>0.023583</td>
          <td>22.865059</td>
          <td>0.041941</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.316386</td>
          <td>0.694087</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.614827</td>
          <td>0.270096</td>
          <td>26.589108</td>
          <td>0.466141</td>
          <td>25.929655</td>
          <td>0.567387</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.223459</td>
          <td>0.337152</td>
          <td>25.745273</td>
          <td>0.082692</td>
          <td>25.416512</td>
          <td>0.055384</td>
          <td>24.886175</td>
          <td>0.056930</td>
          <td>24.318003</td>
          <td>0.065424</td>
          <td>23.907372</td>
          <td>0.102985</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>2.147172</td>
          <td>26.188345</td>
          <td>0.332611</td>
          <td>26.475369</td>
          <td>0.158842</td>
          <td>26.154801</td>
          <td>0.108434</td>
          <td>26.379518</td>
          <td>0.212700</td>
          <td>26.055957</td>
          <td>0.295507</td>
          <td>25.634862</td>
          <td>0.438746</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.857389</td>
          <td>0.216117</td>
          <td>26.596856</td>
          <td>0.156323</td>
          <td>26.634659</td>
          <td>0.258409</td>
          <td>26.111816</td>
          <td>0.304364</td>
          <td>25.130552</td>
          <td>0.291055</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.159876</td>
          <td>0.679728</td>
          <td>28.129384</td>
          <td>0.586036</td>
          <td>26.813312</td>
          <td>0.189469</td>
          <td>26.252817</td>
          <td>0.189694</td>
          <td>26.516993</td>
          <td>0.421282</td>
          <td>25.139292</td>
          <td>0.295503</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.619132</td>
          <td>2.286058</td>
          <td>26.697566</td>
          <td>0.174899</td>
          <td>25.760906</td>
          <td>0.126876</td>
          <td>25.820759</td>
          <td>0.246356</td>
          <td>25.227715</td>
          <td>0.322762</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.639280</td>
          <td>0.467291</td>
          <td>26.355104</td>
          <td>0.141902</td>
          <td>26.125817</td>
          <td>0.104576</td>
          <td>25.409531</td>
          <td>0.091330</td>
          <td>25.646353</td>
          <td>0.208902</td>
          <td>25.041169</td>
          <td>0.272225</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.890625</td>
          <td>26.386398</td>
          <td>0.344997</td>
          <td>26.543849</td>
          <td>0.143835</td>
          <td>26.026148</td>
          <td>0.080698</td>
          <td>25.371003</td>
          <td>0.073772</td>
          <td>24.881528</td>
          <td>0.091443</td>
          <td>24.585881</td>
          <td>0.157168</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.900723</td>
          <td>0.998177</td>
          <td>28.055316</td>
          <td>0.489145</td>
          <td>27.430127</td>
          <td>0.268404</td>
          <td>27.064622</td>
          <td>0.312450</td>
          <td>27.014229</td>
          <td>0.527122</td>
          <td>25.440714</td>
          <td>0.319568</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.895338</td>
          <td>0.245083</td>
          <td>25.885082</td>
          <td>0.087054</td>
          <td>24.761490</td>
          <td>0.028627</td>
          <td>23.847746</td>
          <td>0.021038</td>
          <td>23.141381</td>
          <td>0.021453</td>
          <td>22.821922</td>
          <td>0.036297</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.150302</td>
          <td>1.164373</td>
          <td>27.211399</td>
          <td>0.276450</td>
          <td>26.693317</td>
          <td>0.286883</td>
          <td>26.092845</td>
          <td>0.316423</td>
          <td>24.926360</td>
          <td>0.260849</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.309658</td>
          <td>0.324939</td>
          <td>25.748334</td>
          <td>0.071874</td>
          <td>25.477850</td>
          <td>0.049708</td>
          <td>24.801347</td>
          <td>0.044569</td>
          <td>24.405088</td>
          <td>0.060099</td>
          <td>23.692609</td>
          <td>0.072186</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>2.147172</td>
          <td>26.199275</td>
          <td>0.312776</td>
          <td>26.366627</td>
          <td>0.132095</td>
          <td>26.235588</td>
          <td>0.104940</td>
          <td>26.020125</td>
          <td>0.141252</td>
          <td>25.947096</td>
          <td>0.245639</td>
          <td>26.267664</td>
          <td>0.636503</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.293453</td>
          <td>0.273390</td>
          <td>26.913365</td>
          <td>0.177129</td>
          <td>27.078614</td>
          <td>0.320668</td>
          <td>26.439258</td>
          <td>0.345087</td>
          <td>25.307013</td>
          <td>0.291321</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.568926</td>
          <td>0.409828</td>
          <td>27.170611</td>
          <td>0.253907</td>
          <td>26.862609</td>
          <td>0.175061</td>
          <td>26.358778</td>
          <td>0.183011</td>
          <td>26.160570</td>
          <td>0.284440</td>
          <td>25.137756</td>
          <td>0.261911</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.072423</td>
          <td>0.617518</td>
          <td>28.006645</td>
          <td>0.513202</td>
          <td>26.382955</td>
          <td>0.123646</td>
          <td>25.938152</td>
          <td>0.136537</td>
          <td>25.861762</td>
          <td>0.236790</td>
          <td>25.538017</td>
          <td>0.383295</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.211760</td>
          <td>0.651504</td>
          <td>26.548351</td>
          <td>0.149246</td>
          <td>26.060798</td>
          <td>0.086505</td>
          <td>25.671939</td>
          <td>0.100158</td>
          <td>25.065245</td>
          <td>0.111633</td>
          <td>25.184273</td>
          <td>0.269444</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
