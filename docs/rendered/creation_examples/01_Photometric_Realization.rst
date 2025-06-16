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

    <pzflow.flow.Flow at 0x7fbbb739b790>



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
    0      23.994413  0.275990  0.191010  
    1      25.391064  0.168567  0.137164  
    2      24.304707  0.098191  0.067217  
    3      25.291103  0.147561  0.106891  
    4      25.096743  0.053793  0.049502  
    ...          ...       ...       ...  
    99995  24.737946  0.045177  0.038820  
    99996  24.224169  0.070545  0.042192  
    99997  25.613836  0.054468  0.031315  
    99998  25.274899  0.063498  0.032115  
    99999  25.699642  0.126278  0.111264  
    
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
          <td>1.398944</td>
          <td>27.537430</td>
          <td>0.794307</td>
          <td>26.571551</td>
          <td>0.147283</td>
          <td>25.957736</td>
          <td>0.075958</td>
          <td>25.154473</td>
          <td>0.060888</td>
          <td>24.739478</td>
          <td>0.080680</td>
          <td>23.901548</td>
          <td>0.086676</td>
          <td>0.275990</td>
          <td>0.191010</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.280939</td>
          <td>0.668876</td>
          <td>27.513507</td>
          <td>0.322107</td>
          <td>26.837432</td>
          <td>0.163419</td>
          <td>26.220223</td>
          <td>0.154766</td>
          <td>25.767414</td>
          <td>0.196327</td>
          <td>25.332204</td>
          <td>0.292671</td>
          <td>0.168567</td>
          <td>0.137164</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>33.470815</td>
          <td>6.018902</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.121371</td>
          <td>0.461264</td>
          <td>25.981935</td>
          <td>0.126028</td>
          <td>25.101493</td>
          <td>0.110855</td>
          <td>24.170390</td>
          <td>0.109715</td>
          <td>0.098191</td>
          <td>0.067217</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.162404</td>
          <td>0.616003</td>
          <td>28.385169</td>
          <td>0.620171</td>
          <td>27.109294</td>
          <td>0.205672</td>
          <td>26.408869</td>
          <td>0.181743</td>
          <td>25.603557</td>
          <td>0.170913</td>
          <td>25.768830</td>
          <td>0.412679</td>
          <td>0.147561</td>
          <td>0.106891</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.045935</td>
          <td>0.567144</td>
          <td>26.181031</td>
          <td>0.105001</td>
          <td>25.962984</td>
          <td>0.076311</td>
          <td>25.597805</td>
          <td>0.090097</td>
          <td>25.567390</td>
          <td>0.165729</td>
          <td>25.115913</td>
          <td>0.245358</td>
          <td>0.053793</td>
          <td>0.049502</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.536645</td>
          <td>0.142931</td>
          <td>25.441748</td>
          <td>0.048070</td>
          <td>25.095004</td>
          <td>0.057758</td>
          <td>24.923481</td>
          <td>0.094863</td>
          <td>24.449893</td>
          <td>0.139825</td>
          <td>0.045177</td>
          <td>0.038820</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.878428</td>
          <td>0.191234</td>
          <td>26.124742</td>
          <td>0.088011</td>
          <td>25.347765</td>
          <td>0.072261</td>
          <td>24.698909</td>
          <td>0.077842</td>
          <td>24.503073</td>
          <td>0.146374</td>
          <td>0.070545</td>
          <td>0.042192</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.106175</td>
          <td>0.275705</td>
          <td>26.551627</td>
          <td>0.144784</td>
          <td>26.345575</td>
          <td>0.106821</td>
          <td>26.113487</td>
          <td>0.141205</td>
          <td>26.457966</td>
          <td>0.345085</td>
          <td>25.211060</td>
          <td>0.265266</td>
          <td>0.054468</td>
          <td>0.031315</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.160655</td>
          <td>0.615247</td>
          <td>26.115733</td>
          <td>0.099173</td>
          <td>26.084956</td>
          <td>0.084981</td>
          <td>25.796108</td>
          <td>0.107205</td>
          <td>26.008052</td>
          <td>0.239935</td>
          <td>24.922361</td>
          <td>0.208937</td>
          <td>0.063498</td>
          <td>0.032115</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.744076</td>
          <td>0.454366</td>
          <td>26.751283</td>
          <td>0.171726</td>
          <td>26.555669</td>
          <td>0.128249</td>
          <td>26.110870</td>
          <td>0.140887</td>
          <td>25.764293</td>
          <td>0.195812</td>
          <td>25.753512</td>
          <td>0.407861</td>
          <td>0.126278</td>
          <td>0.111264</td>
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
          <td>26.995751</td>
          <td>0.664799</td>
          <td>26.742763</td>
          <td>0.224698</td>
          <td>26.032601</td>
          <td>0.111791</td>
          <td>25.215013</td>
          <td>0.089752</td>
          <td>24.702442</td>
          <td>0.107591</td>
          <td>24.224556</td>
          <td>0.159027</td>
          <td>0.275990</td>
          <td>0.191010</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.689570</td>
          <td>0.505818</td>
          <td>27.627443</td>
          <td>0.424164</td>
          <td>26.652006</td>
          <td>0.175360</td>
          <td>26.722924</td>
          <td>0.296776</td>
          <td>25.508234</td>
          <td>0.197789</td>
          <td>25.109439</td>
          <td>0.305586</td>
          <td>0.168567</td>
          <td>0.137164</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.843910</td>
          <td>1.634219</td>
          <td>27.988277</td>
          <td>0.488626</td>
          <td>26.065449</td>
          <td>0.163543</td>
          <td>25.080864</td>
          <td>0.130734</td>
          <td>24.528157</td>
          <td>0.180038</td>
          <td>0.098191</td>
          <td>0.067217</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.219209</td>
          <td>0.724847</td>
          <td>28.820117</td>
          <td>0.952565</td>
          <td>27.508955</td>
          <td>0.347478</td>
          <td>26.486184</td>
          <td>0.239813</td>
          <td>25.129522</td>
          <td>0.140385</td>
          <td>25.746125</td>
          <td>0.490483</td>
          <td>0.147561</td>
          <td>0.106891</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.218205</td>
          <td>0.337753</td>
          <td>26.085065</td>
          <td>0.112219</td>
          <td>25.921000</td>
          <td>0.087286</td>
          <td>25.729942</td>
          <td>0.120716</td>
          <td>25.287722</td>
          <td>0.154022</td>
          <td>24.698637</td>
          <td>0.204912</td>
          <td>0.053793</td>
          <td>0.049502</td>
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
          <td>26.546261</td>
          <td>0.434541</td>
          <td>26.309580</td>
          <td>0.135956</td>
          <td>25.391432</td>
          <td>0.054476</td>
          <td>24.959309</td>
          <td>0.061105</td>
          <td>24.789626</td>
          <td>0.099710</td>
          <td>25.200401</td>
          <td>0.308393</td>
          <td>0.045177</td>
          <td>0.038820</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.845517</td>
          <td>0.544289</td>
          <td>26.836187</td>
          <td>0.213635</td>
          <td>26.173745</td>
          <td>0.109186</td>
          <td>25.106417</td>
          <td>0.069998</td>
          <td>24.777128</td>
          <td>0.099151</td>
          <td>23.969479</td>
          <td>0.109956</td>
          <td>0.070545</td>
          <td>0.042192</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.576137</td>
          <td>0.444664</td>
          <td>26.264203</td>
          <td>0.130810</td>
          <td>26.230616</td>
          <td>0.114204</td>
          <td>26.096410</td>
          <td>0.165109</td>
          <td>25.961359</td>
          <td>0.270112</td>
          <td>26.566641</td>
          <td>0.834257</td>
          <td>0.054468</td>
          <td>0.031315</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.437726</td>
          <td>0.400658</td>
          <td>26.256327</td>
          <td>0.130138</td>
          <td>26.169232</td>
          <td>0.108452</td>
          <td>25.699063</td>
          <td>0.117459</td>
          <td>25.784374</td>
          <td>0.233983</td>
          <td>24.919417</td>
          <td>0.246042</td>
          <td>0.063498</td>
          <td>0.032115</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.442192</td>
          <td>0.412270</td>
          <td>27.040589</td>
          <td>0.260509</td>
          <td>26.813869</td>
          <td>0.195679</td>
          <td>26.197062</td>
          <td>0.187027</td>
          <td>25.795904</td>
          <td>0.244669</td>
          <td>26.336069</td>
          <td>0.739148</td>
          <td>0.126278</td>
          <td>0.111264</td>
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

.. parsed-literal::

    




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
          <td>27.897018</td>
          <td>1.249805</td>
          <td>27.136549</td>
          <td>0.344488</td>
          <td>26.009517</td>
          <td>0.124010</td>
          <td>25.053856</td>
          <td>0.088525</td>
          <td>24.610120</td>
          <td>0.112392</td>
          <td>23.950126</td>
          <td>0.142284</td>
          <td>0.275990</td>
          <td>0.191010</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.124966</td>
          <td>1.276176</td>
          <td>27.528983</td>
          <td>0.395965</td>
          <td>26.504558</td>
          <td>0.155891</td>
          <td>26.460087</td>
          <td>0.241424</td>
          <td>25.962670</td>
          <td>0.289964</td>
          <td>25.330196</td>
          <td>0.366711</td>
          <td>0.168567</td>
          <td>0.137164</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.727050</td>
          <td>0.472893</td>
          <td>30.037853</td>
          <td>1.708049</td>
          <td>27.862751</td>
          <td>0.408565</td>
          <td>25.873642</td>
          <td>0.125487</td>
          <td>24.980975</td>
          <td>0.108771</td>
          <td>24.287390</td>
          <td>0.132751</td>
          <td>0.098191</td>
          <td>0.067217</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.606106</td>
          <td>0.815143</td>
          <td>27.342098</td>
          <td>0.295439</td>
          <td>26.109304</td>
          <td>0.169319</td>
          <td>25.358603</td>
          <td>0.165540</td>
          <td>25.519246</td>
          <td>0.401616</td>
          <td>0.147561</td>
          <td>0.106891</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.294326</td>
          <td>0.327902</td>
          <td>26.059623</td>
          <td>0.097337</td>
          <td>25.776831</td>
          <td>0.067049</td>
          <td>25.874406</td>
          <td>0.119036</td>
          <td>25.542582</td>
          <td>0.167859</td>
          <td>25.413304</td>
          <td>0.322845</td>
          <td>0.053793</td>
          <td>0.049502</td>
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
          <td>26.329822</td>
          <td>0.334800</td>
          <td>26.577955</td>
          <td>0.151084</td>
          <td>25.432840</td>
          <td>0.048842</td>
          <td>25.051685</td>
          <td>0.056985</td>
          <td>24.843252</td>
          <td>0.090510</td>
          <td>24.798240</td>
          <td>0.192703</td>
          <td>0.045177</td>
          <td>0.038820</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.598516</td>
          <td>0.156409</td>
          <td>25.998813</td>
          <td>0.082281</td>
          <td>25.313135</td>
          <td>0.073372</td>
          <td>24.946829</td>
          <td>0.101113</td>
          <td>24.166309</td>
          <td>0.114330</td>
          <td>0.070545</td>
          <td>0.042192</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.645178</td>
          <td>0.428267</td>
          <td>26.840204</td>
          <td>0.189226</td>
          <td>26.382799</td>
          <td>0.113230</td>
          <td>26.339569</td>
          <td>0.175929</td>
          <td>26.112254</td>
          <td>0.267748</td>
          <td>26.077652</td>
          <td>0.531834</td>
          <td>0.054468</td>
          <td>0.031315</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.480287</td>
          <td>0.378943</td>
          <td>25.997292</td>
          <td>0.091991</td>
          <td>26.110786</td>
          <td>0.089848</td>
          <td>25.663789</td>
          <td>0.098817</td>
          <td>25.867449</td>
          <td>0.220250</td>
          <td>26.000024</td>
          <td>0.505463</td>
          <td>0.063498</td>
          <td>0.032115</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.599722</td>
          <td>0.173358</td>
          <td>26.458484</td>
          <td>0.138224</td>
          <td>26.402859</td>
          <td>0.212435</td>
          <td>26.302908</td>
          <td>0.352889</td>
          <td>27.117979</td>
          <td>1.158708</td>
          <td>0.126278</td>
          <td>0.111264</td>
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
