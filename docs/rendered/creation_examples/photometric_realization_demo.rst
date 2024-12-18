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

    <pzflow.flow.Flow at 0x7f51848296f0>



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
          <td>26.941372</td>
          <td>0.201627</td>
          <td>26.182139</td>
          <td>0.092568</td>
          <td>25.397481</td>
          <td>0.075509</td>
          <td>25.059142</td>
          <td>0.106831</td>
          <td>25.091633</td>
          <td>0.240496</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.075079</td>
          <td>0.496008</td>
          <td>27.496212</td>
          <td>0.282965</td>
          <td>28.650046</td>
          <td>0.965784</td>
          <td>27.077186</td>
          <td>0.551333</td>
          <td>27.149512</td>
          <td>1.057601</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.901708</td>
          <td>0.510810</td>
          <td>25.922797</td>
          <td>0.083722</td>
          <td>24.734561</td>
          <td>0.025766</td>
          <td>23.856727</td>
          <td>0.019503</td>
          <td>23.176580</td>
          <td>0.020406</td>
          <td>22.782803</td>
          <td>0.032192</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.671253</td>
          <td>0.865753</td>
          <td>27.487145</td>
          <td>0.315407</td>
          <td>27.436423</td>
          <td>0.269548</td>
          <td>26.618764</td>
          <td>0.216804</td>
          <td>25.924596</td>
          <td>0.223909</td>
          <td>26.460179</td>
          <td>0.681877</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.226390</td>
          <td>0.303777</td>
          <td>25.800861</td>
          <td>0.075192</td>
          <td>25.331021</td>
          <td>0.043571</td>
          <td>24.924206</td>
          <td>0.049631</td>
          <td>24.431204</td>
          <td>0.061419</td>
          <td>23.793529</td>
          <td>0.078802</td>
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
          <td>26.370920</td>
          <td>0.340785</td>
          <td>26.248430</td>
          <td>0.111360</td>
          <td>26.060311</td>
          <td>0.083155</td>
          <td>25.837573</td>
          <td>0.111157</td>
          <td>25.952796</td>
          <td>0.229213</td>
          <td>25.855171</td>
          <td>0.440715</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.985276</td>
          <td>0.209179</td>
          <td>27.353960</td>
          <td>0.251965</td>
          <td>26.348531</td>
          <td>0.172674</td>
          <td>25.908127</td>
          <td>0.220862</td>
          <td>26.531131</td>
          <td>0.715536</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.165276</td>
          <td>1.164259</td>
          <td>27.201790</td>
          <td>0.250301</td>
          <td>27.164582</td>
          <td>0.215403</td>
          <td>26.787182</td>
          <td>0.249250</td>
          <td>25.551406</td>
          <td>0.163485</td>
          <td>25.588709</td>
          <td>0.358917</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.118012</td>
          <td>0.233598</td>
          <td>26.435599</td>
          <td>0.115548</td>
          <td>25.925066</td>
          <td>0.119957</td>
          <td>25.719243</td>
          <td>0.188517</td>
          <td>25.430090</td>
          <td>0.316588</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.727791</td>
          <td>0.897167</td>
          <td>26.484076</td>
          <td>0.136603</td>
          <td>25.984831</td>
          <td>0.077798</td>
          <td>25.750233</td>
          <td>0.102989</td>
          <td>25.387285</td>
          <td>0.142027</td>
          <td>24.689416</td>
          <td>0.171657</td>
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
          <td>27.982032</td>
          <td>1.131745</td>
          <td>26.493621</td>
          <td>0.158388</td>
          <td>26.127649</td>
          <td>0.103701</td>
          <td>25.361474</td>
          <td>0.086643</td>
          <td>25.041295</td>
          <td>0.123442</td>
          <td>24.879670</td>
          <td>0.236163</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.697057</td>
          <td>1.644303</td>
          <td>27.843712</td>
          <td>0.471362</td>
          <td>27.488737</td>
          <td>0.326037</td>
          <td>26.759632</td>
          <td>0.284996</td>
          <td>26.201858</td>
          <td>0.325885</td>
          <td>25.820828</td>
          <td>0.495174</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.516578</td>
          <td>0.429480</td>
          <td>26.129916</td>
          <td>0.118084</td>
          <td>24.788052</td>
          <td>0.032467</td>
          <td>23.869952</td>
          <td>0.023802</td>
          <td>23.128885</td>
          <td>0.023467</td>
          <td>22.830795</td>
          <td>0.040688</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.799651</td>
          <td>1.772106</td>
          <td>28.620001</td>
          <td>0.848028</td>
          <td>28.076983</td>
          <td>0.540819</td>
          <td>26.304291</td>
          <td>0.208996</td>
          <td>26.801937</td>
          <td>0.545217</td>
          <td>25.108772</td>
          <td>0.303426</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.080649</td>
          <td>0.300910</td>
          <td>25.800951</td>
          <td>0.086841</td>
          <td>25.459526</td>
          <td>0.057538</td>
          <td>24.864318</td>
          <td>0.055836</td>
          <td>24.351612</td>
          <td>0.067401</td>
          <td>23.694831</td>
          <td>0.085459</td>
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
          <td>26.530175</td>
          <td>0.433548</td>
          <td>26.239591</td>
          <td>0.129705</td>
          <td>26.064299</td>
          <td>0.100184</td>
          <td>26.542293</td>
          <td>0.243457</td>
          <td>25.157170</td>
          <td>0.139303</td>
          <td>25.526250</td>
          <td>0.403853</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.253930</td>
          <td>0.720839</td>
          <td>27.116776</td>
          <td>0.267651</td>
          <td>27.016793</td>
          <td>0.222844</td>
          <td>26.454387</td>
          <td>0.222688</td>
          <td>26.554105</td>
          <td>0.430139</td>
          <td>25.115526</td>
          <td>0.287544</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.105584</td>
          <td>1.220746</td>
          <td>27.054465</td>
          <td>0.256212</td>
          <td>26.441674</td>
          <td>0.137967</td>
          <td>26.514752</td>
          <td>0.236095</td>
          <td>26.225981</td>
          <td>0.335971</td>
          <td>25.272098</td>
          <td>0.328621</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.595895</td>
          <td>1.587013</td>
          <td>27.598290</td>
          <td>0.400971</td>
          <td>26.624176</td>
          <td>0.164311</td>
          <td>25.865556</td>
          <td>0.138887</td>
          <td>25.498272</td>
          <td>0.188267</td>
          <td>25.137904</td>
          <td>0.300387</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.280378</td>
          <td>0.354943</td>
          <td>26.622934</td>
          <td>0.178366</td>
          <td>26.114685</td>
          <td>0.103563</td>
          <td>25.758048</td>
          <td>0.123830</td>
          <td>25.007876</td>
          <td>0.121107</td>
          <td>25.190418</td>
          <td>0.307102</td>
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
          <td>26.233244</td>
          <td>0.305475</td>
          <td>26.576995</td>
          <td>0.147990</td>
          <td>26.052430</td>
          <td>0.082590</td>
          <td>25.280323</td>
          <td>0.068083</td>
          <td>25.071463</td>
          <td>0.108001</td>
          <td>24.923147</td>
          <td>0.209102</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.441890</td>
          <td>0.304415</td>
          <td>27.472180</td>
          <td>0.277746</td>
          <td>26.668051</td>
          <td>0.226093</td>
          <td>26.867207</td>
          <td>0.472927</td>
          <td>25.228915</td>
          <td>0.269403</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.642188</td>
          <td>0.442447</td>
          <td>25.853414</td>
          <td>0.084663</td>
          <td>24.751762</td>
          <td>0.028385</td>
          <td>23.911747</td>
          <td>0.022221</td>
          <td>23.135612</td>
          <td>0.021347</td>
          <td>22.820159</td>
          <td>0.036241</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.951803</td>
          <td>1.037982</td>
          <td>27.022132</td>
          <td>0.236732</td>
          <td>27.092964</td>
          <td>0.393541</td>
          <td>25.846462</td>
          <td>0.259271</td>
          <td>25.608316</td>
          <td>0.446478</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.893029</td>
          <td>0.507981</td>
          <td>25.867873</td>
          <td>0.079867</td>
          <td>25.459356</td>
          <td>0.048898</td>
          <td>24.760412</td>
          <td>0.042980</td>
          <td>24.404601</td>
          <td>0.060073</td>
          <td>23.658652</td>
          <td>0.070050</td>
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
          <td>27.457406</td>
          <td>0.785779</td>
          <td>26.453262</td>
          <td>0.142339</td>
          <td>26.212045</td>
          <td>0.102801</td>
          <td>26.056066</td>
          <td>0.145690</td>
          <td>26.044866</td>
          <td>0.266136</td>
          <td>25.250390</td>
          <td>0.295273</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.485519</td>
          <td>0.376614</td>
          <td>27.122636</td>
          <td>0.237673</td>
          <td>26.701810</td>
          <td>0.147856</td>
          <td>26.386385</td>
          <td>0.181310</td>
          <td>25.784002</td>
          <td>0.202239</td>
          <td>25.837257</td>
          <td>0.441231</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.758694</td>
          <td>0.473045</td>
          <td>27.455602</td>
          <td>0.319727</td>
          <td>27.308667</td>
          <td>0.254116</td>
          <td>26.360313</td>
          <td>0.183249</td>
          <td>25.819873</td>
          <td>0.214932</td>
          <td>25.170494</td>
          <td>0.269004</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.237303</td>
          <td>1.275846</td>
          <td>27.139400</td>
          <td>0.261283</td>
          <td>26.803361</td>
          <td>0.177390</td>
          <td>25.845838</td>
          <td>0.126058</td>
          <td>25.059692</td>
          <td>0.119750</td>
          <td>24.724765</td>
          <td>0.198260</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.607327</td>
          <td>0.419408</td>
          <td>26.455975</td>
          <td>0.137849</td>
          <td>26.210992</td>
          <td>0.098707</td>
          <td>25.745165</td>
          <td>0.106785</td>
          <td>25.227640</td>
          <td>0.128555</td>
          <td>25.385331</td>
          <td>0.316887</td>
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
