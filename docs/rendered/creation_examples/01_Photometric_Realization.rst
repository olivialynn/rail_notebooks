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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f62711ee9e0>



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
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.038418  0.036111  
    1      25.391064  0.014367  0.011681  
    2      24.304707  0.095977  0.095369  
    3      25.291103  0.094239  0.060293  
    4      25.096743  0.091332  0.047744  
    ...          ...       ...       ...  
    99995  24.737946  0.138535  0.138166  
    99996  24.224169  0.024182  0.015491  
    99997  25.613836  0.016112  0.012539  
    99998  25.274899  0.005096  0.003083  
    99999  25.699642  0.099944  0.077624  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.688129</td>
          <td>0.162736</td>
          <td>25.985851</td>
          <td>0.077868</td>
          <td>25.149211</td>
          <td>0.060604</td>
          <td>24.703380</td>
          <td>0.078150</td>
          <td>24.106071</td>
          <td>0.103719</td>
          <td>0.038418</td>
          <td>0.036111</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.543257</td>
          <td>2.235396</td>
          <td>27.302253</td>
          <td>0.271728</td>
          <td>26.555035</td>
          <td>0.128179</td>
          <td>26.099280</td>
          <td>0.139487</td>
          <td>26.403653</td>
          <td>0.330574</td>
          <td>25.783037</td>
          <td>0.417189</td>
          <td>0.014367</td>
          <td>0.011681</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.430781</td>
          <td>0.740314</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.965743</td>
          <td>0.124270</td>
          <td>25.027838</td>
          <td>0.103947</td>
          <td>24.623485</td>
          <td>0.162280</td>
          <td>0.095977</td>
          <td>0.095369</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>31.142839</td>
          <td>3.722489</td>
          <td>30.451627</td>
          <td>1.974973</td>
          <td>27.528208</td>
          <td>0.290384</td>
          <td>26.158851</td>
          <td>0.146826</td>
          <td>25.364067</td>
          <td>0.139213</td>
          <td>25.524687</td>
          <td>0.341285</td>
          <td>0.094239</td>
          <td>0.060293</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.184346</td>
          <td>0.293687</td>
          <td>26.029863</td>
          <td>0.091983</td>
          <td>25.900568</td>
          <td>0.072214</td>
          <td>25.755778</td>
          <td>0.103490</td>
          <td>25.552050</td>
          <td>0.163575</td>
          <td>25.076900</td>
          <td>0.237588</td>
          <td>0.091332</td>
          <td>0.047744</td>
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
          <td>0.389450</td>
          <td>27.010659</td>
          <td>0.552943</td>
          <td>26.240339</td>
          <td>0.110577</td>
          <td>25.433425</td>
          <td>0.047716</td>
          <td>25.018367</td>
          <td>0.053959</td>
          <td>25.005440</td>
          <td>0.101929</td>
          <td>25.153822</td>
          <td>0.253125</td>
          <td>0.138535</td>
          <td>0.138166</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.537832</td>
          <td>0.388245</td>
          <td>26.666687</td>
          <td>0.159785</td>
          <td>26.151520</td>
          <td>0.090109</td>
          <td>25.269236</td>
          <td>0.067408</td>
          <td>24.690141</td>
          <td>0.077242</td>
          <td>24.207031</td>
          <td>0.113278</td>
          <td>0.024182</td>
          <td>0.015491</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.762412</td>
          <td>0.460662</td>
          <td>26.885172</td>
          <td>0.192324</td>
          <td>26.366524</td>
          <td>0.108794</td>
          <td>26.270073</td>
          <td>0.161507</td>
          <td>25.739941</td>
          <td>0.191837</td>
          <td>25.462681</td>
          <td>0.324918</td>
          <td>0.016112</td>
          <td>0.012539</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.684832</td>
          <td>0.194639</td>
          <td>26.215226</td>
          <td>0.108182</td>
          <td>26.027829</td>
          <td>0.080807</td>
          <td>25.787459</td>
          <td>0.106398</td>
          <td>25.609648</td>
          <td>0.171800</td>
          <td>25.758620</td>
          <td>0.409463</td>
          <td>0.005096</td>
          <td>0.003083</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.431002</td>
          <td>0.357269</td>
          <td>26.857847</td>
          <td>0.187944</td>
          <td>26.431372</td>
          <td>0.115124</td>
          <td>26.233284</td>
          <td>0.156506</td>
          <td>27.021851</td>
          <td>0.529644</td>
          <td>26.106509</td>
          <td>0.531152</td>
          <td>0.099944</td>
          <td>0.077624</td>
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
          <td>1.398944</td>
          <td>28.097845</td>
          <td>1.210802</td>
          <td>26.464248</td>
          <td>0.155097</td>
          <td>26.065283</td>
          <td>0.098651</td>
          <td>25.240744</td>
          <td>0.078277</td>
          <td>24.617468</td>
          <td>0.085608</td>
          <td>24.101361</td>
          <td>0.122496</td>
          <td>0.038418</td>
          <td>0.036111</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.834288</td>
          <td>0.211336</td>
          <td>26.711155</td>
          <td>0.171727</td>
          <td>26.133050</td>
          <td>0.169307</td>
          <td>26.363303</td>
          <td>0.370185</td>
          <td>25.046717</td>
          <td>0.271000</td>
          <td>0.014367</td>
          <td>0.011681</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.788306</td>
          <td>0.853926</td>
          <td>26.192984</td>
          <td>0.183493</td>
          <td>24.970003</td>
          <td>0.119547</td>
          <td>24.605086</td>
          <td>0.193409</td>
          <td>0.095977</td>
          <td>0.095369</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.673389</td>
          <td>1.639766</td>
          <td>28.387536</td>
          <td>0.705110</td>
          <td>27.153235</td>
          <td>0.253353</td>
          <td>26.652678</td>
          <td>0.266461</td>
          <td>25.680404</td>
          <td>0.217128</td>
          <td>25.426908</td>
          <td>0.373897</td>
          <td>0.094239</td>
          <td>0.060293</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.709538</td>
          <td>0.494766</td>
          <td>26.081605</td>
          <td>0.112753</td>
          <td>26.067724</td>
          <td>0.100148</td>
          <td>25.795985</td>
          <td>0.128965</td>
          <td>25.315330</td>
          <td>0.159036</td>
          <td>26.227716</td>
          <td>0.671658</td>
          <td>0.091332</td>
          <td>0.047744</td>
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
          <td>0.389450</td>
          <td>26.338434</td>
          <td>0.384597</td>
          <td>26.278491</td>
          <td>0.139030</td>
          <td>25.303661</td>
          <td>0.053292</td>
          <td>25.101361</td>
          <td>0.073398</td>
          <td>24.916094</td>
          <td>0.117620</td>
          <td>25.521105</td>
          <td>0.416962</td>
          <td>0.138535</td>
          <td>0.138166</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.649852</td>
          <td>0.929380</td>
          <td>26.531273</td>
          <td>0.163757</td>
          <td>26.125603</td>
          <td>0.103657</td>
          <td>25.133899</td>
          <td>0.070980</td>
          <td>24.689813</td>
          <td>0.090931</td>
          <td>24.255602</td>
          <td>0.139511</td>
          <td>0.024182</td>
          <td>0.015491</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.743564</td>
          <td>0.983875</td>
          <td>26.592186</td>
          <td>0.172363</td>
          <td>26.288752</td>
          <td>0.119422</td>
          <td>26.253219</td>
          <td>0.187488</td>
          <td>26.388316</td>
          <td>0.377507</td>
          <td>25.464816</td>
          <td>0.378121</td>
          <td>0.016112</td>
          <td>0.012539</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.964792</td>
          <td>0.273990</td>
          <td>26.100456</td>
          <td>0.112823</td>
          <td>26.156745</td>
          <td>0.106376</td>
          <td>26.081771</td>
          <td>0.161980</td>
          <td>25.731503</td>
          <td>0.222161</td>
          <td>25.185369</td>
          <td>0.303003</td>
          <td>0.005096</td>
          <td>0.003083</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.961594</td>
          <td>0.597055</td>
          <td>27.022535</td>
          <td>0.252494</td>
          <td>26.559995</td>
          <td>0.154795</td>
          <td>26.554144</td>
          <td>0.247151</td>
          <td>26.119085</td>
          <td>0.312422</td>
          <td>27.165026</td>
          <td>1.210547</td>
          <td>0.099944</td>
          <td>0.077624</td>
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
          <td>1.398944</td>
          <td>27.213812</td>
          <td>0.645083</td>
          <td>27.094183</td>
          <td>0.232537</td>
          <td>26.235374</td>
          <td>0.098808</td>
          <td>25.222084</td>
          <td>0.065933</td>
          <td>24.666999</td>
          <td>0.077102</td>
          <td>23.981628</td>
          <td>0.094808</td>
          <td>0.038418</td>
          <td>0.036111</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.345671</td>
          <td>0.282002</td>
          <td>26.713552</td>
          <td>0.147303</td>
          <td>26.080106</td>
          <td>0.137528</td>
          <td>26.388224</td>
          <td>0.327234</td>
          <td>25.604125</td>
          <td>0.364057</td>
          <td>0.014367</td>
          <td>0.011681</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>32.062405</td>
          <td>4.710482</td>
          <td>30.096819</td>
          <td>1.774261</td>
          <td>28.211013</td>
          <td>0.540714</td>
          <td>26.186322</td>
          <td>0.168221</td>
          <td>25.058237</td>
          <td>0.119119</td>
          <td>24.258195</td>
          <td>0.132613</td>
          <td>0.095977</td>
          <td>0.095369</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.185778</td>
          <td>1.219571</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.392229</td>
          <td>0.279316</td>
          <td>26.212082</td>
          <td>0.166270</td>
          <td>25.303757</td>
          <td>0.142568</td>
          <td>24.650096</td>
          <td>0.179384</td>
          <td>0.094239</td>
          <td>0.060293</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.880575</td>
          <td>0.239476</td>
          <td>26.086701</td>
          <td>0.102420</td>
          <td>26.012218</td>
          <td>0.085157</td>
          <td>25.607431</td>
          <td>0.097349</td>
          <td>25.387586</td>
          <td>0.151512</td>
          <td>25.099920</td>
          <td>0.258106</td>
          <td>0.091332</td>
          <td>0.047744</td>
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
          <td>0.389450</td>
          <td>26.444305</td>
          <td>0.413002</td>
          <td>26.269603</td>
          <td>0.136135</td>
          <td>25.575532</td>
          <td>0.066794</td>
          <td>25.169158</td>
          <td>0.076725</td>
          <td>24.673228</td>
          <td>0.093723</td>
          <td>24.661819</td>
          <td>0.206050</td>
          <td>0.138535</td>
          <td>0.138166</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.814807</td>
          <td>0.182076</td>
          <td>25.886166</td>
          <td>0.071701</td>
          <td>25.051767</td>
          <td>0.055913</td>
          <td>24.867974</td>
          <td>0.090852</td>
          <td>24.357448</td>
          <td>0.129830</td>
          <td>0.024182</td>
          <td>0.015491</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.030393</td>
          <td>0.561751</td>
          <td>26.746038</td>
          <td>0.171368</td>
          <td>26.401716</td>
          <td>0.112502</td>
          <td>26.624812</td>
          <td>0.218513</td>
          <td>25.974507</td>
          <td>0.233994</td>
          <td>25.743095</td>
          <td>0.405651</td>
          <td>0.016112</td>
          <td>0.012539</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.896981</td>
          <td>0.232322</td>
          <td>26.279196</td>
          <td>0.114407</td>
          <td>26.156950</td>
          <td>0.090562</td>
          <td>25.790575</td>
          <td>0.106715</td>
          <td>26.276268</td>
          <td>0.298646</td>
          <td>25.546962</td>
          <td>0.347412</td>
          <td>0.005096</td>
          <td>0.003083</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>29.599087</td>
          <td>2.354079</td>
          <td>26.975944</td>
          <td>0.225134</td>
          <td>26.707461</td>
          <td>0.160799</td>
          <td>25.993412</td>
          <td>0.140720</td>
          <td>25.882982</td>
          <td>0.237140</td>
          <td>25.290813</td>
          <td>0.310483</td>
          <td>0.099944</td>
          <td>0.077624</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
