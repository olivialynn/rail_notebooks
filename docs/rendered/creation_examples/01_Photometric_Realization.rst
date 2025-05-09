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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7ff458699ba0>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.736723</td>
          <td>0.169614</td>
          <td>26.027976</td>
          <td>0.080818</td>
          <td>25.215090</td>
          <td>0.064250</td>
          <td>24.767059</td>
          <td>0.082666</td>
          <td>24.242818</td>
          <td>0.116864</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.755309</td>
          <td>0.389431</td>
          <td>26.634447</td>
          <td>0.137289</td>
          <td>26.183729</td>
          <td>0.149998</td>
          <td>26.208687</td>
          <td>0.282726</td>
          <td>25.058085</td>
          <td>0.233920</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.400203</td>
          <td>0.626736</td>
          <td>29.667681</td>
          <td>1.262372</td>
          <td>25.829130</td>
          <td>0.110341</td>
          <td>25.065070</td>
          <td>0.107386</td>
          <td>24.302763</td>
          <td>0.123114</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.337603</td>
          <td>2.058273</td>
          <td>28.435291</td>
          <td>0.642257</td>
          <td>27.219262</td>
          <td>0.225436</td>
          <td>26.179120</td>
          <td>0.149405</td>
          <td>25.425056</td>
          <td>0.146718</td>
          <td>25.147051</td>
          <td>0.251722</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.125744</td>
          <td>0.280113</td>
          <td>26.114597</td>
          <td>0.099075</td>
          <td>25.936493</td>
          <td>0.074545</td>
          <td>25.702070</td>
          <td>0.098735</td>
          <td>25.366845</td>
          <td>0.139547</td>
          <td>25.399795</td>
          <td>0.309012</td>
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
          <td>0.389450</td>
          <td>28.307040</td>
          <td>1.259532</td>
          <td>26.377015</td>
          <td>0.124526</td>
          <td>25.444710</td>
          <td>0.048197</td>
          <td>25.242027</td>
          <td>0.065803</td>
          <td>24.810251</td>
          <td>0.085873</td>
          <td>25.083573</td>
          <td>0.238902</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.077509</td>
          <td>0.580089</td>
          <td>26.501867</td>
          <td>0.138714</td>
          <td>26.104369</td>
          <td>0.086447</td>
          <td>25.116758</td>
          <td>0.058884</td>
          <td>24.832774</td>
          <td>0.087593</td>
          <td>24.288489</td>
          <td>0.121598</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.576267</td>
          <td>0.399925</td>
          <td>26.882157</td>
          <td>0.191836</td>
          <td>26.390298</td>
          <td>0.111075</td>
          <td>26.265603</td>
          <td>0.160892</td>
          <td>25.705204</td>
          <td>0.186295</td>
          <td>25.875438</td>
          <td>0.447516</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.189478</td>
          <td>0.294903</td>
          <td>26.200033</td>
          <td>0.106757</td>
          <td>25.929290</td>
          <td>0.074072</td>
          <td>25.934683</td>
          <td>0.120964</td>
          <td>25.738666</td>
          <td>0.191631</td>
          <td>25.093610</td>
          <td>0.240889</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.843862</td>
          <td>0.489484</td>
          <td>27.117640</td>
          <td>0.233526</td>
          <td>26.720191</td>
          <td>0.147809</td>
          <td>26.117315</td>
          <td>0.141672</td>
          <td>26.029264</td>
          <td>0.244169</td>
          <td>25.590706</td>
          <td>0.359479</td>
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
          <td>1.398944</td>
          <td>28.861345</td>
          <td>1.774364</td>
          <td>26.412144</td>
          <td>0.147715</td>
          <td>26.081168</td>
          <td>0.099567</td>
          <td>25.250957</td>
          <td>0.078602</td>
          <td>24.787986</td>
          <td>0.098975</td>
          <td>23.902748</td>
          <td>0.102535</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.057783</td>
          <td>1.181481</td>
          <td>27.701329</td>
          <td>0.423355</td>
          <td>26.579545</td>
          <td>0.153428</td>
          <td>26.180066</td>
          <td>0.176147</td>
          <td>25.767631</td>
          <td>0.228966</td>
          <td>25.386131</td>
          <td>0.355429</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.916319</td>
          <td>0.462658</td>
          <td>25.674892</td>
          <td>0.116653</td>
          <td>24.818404</td>
          <td>0.103930</td>
          <td>24.310996</td>
          <td>0.149431</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.129991</td>
          <td>1.270017</td>
          <td>27.987068</td>
          <td>0.550943</td>
          <td>28.321691</td>
          <td>0.643452</td>
          <td>26.482990</td>
          <td>0.242435</td>
          <td>25.770440</td>
          <td>0.244397</td>
          <td>25.188791</td>
          <td>0.323467</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.286302</td>
          <td>0.354249</td>
          <td>26.102559</td>
          <td>0.113060</td>
          <td>26.045716</td>
          <td>0.096552</td>
          <td>25.538919</td>
          <td>0.101283</td>
          <td>25.374542</td>
          <td>0.164494</td>
          <td>25.527516</td>
          <td>0.396800</td>
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
          <td>0.389450</td>
          <td>27.424277</td>
          <td>0.814444</td>
          <td>26.294054</td>
          <td>0.135950</td>
          <td>25.415714</td>
          <td>0.056516</td>
          <td>25.148677</td>
          <td>0.073401</td>
          <td>24.878646</td>
          <td>0.109403</td>
          <td>24.899673</td>
          <td>0.245004</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.523398</td>
          <td>0.426558</td>
          <td>26.712473</td>
          <td>0.191404</td>
          <td>25.958423</td>
          <td>0.089772</td>
          <td>25.318461</td>
          <td>0.083785</td>
          <td>24.690024</td>
          <td>0.091203</td>
          <td>24.265956</td>
          <td>0.141159</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.692402</td>
          <td>0.487003</td>
          <td>26.881875</td>
          <td>0.222199</td>
          <td>26.238390</td>
          <td>0.115683</td>
          <td>26.072051</td>
          <td>0.162716</td>
          <td>26.032247</td>
          <td>0.287731</td>
          <td>25.696924</td>
          <td>0.456462</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.031317</td>
          <td>0.295583</td>
          <td>26.037534</td>
          <td>0.109860</td>
          <td>26.054858</td>
          <td>0.100397</td>
          <td>25.985073</td>
          <td>0.153912</td>
          <td>25.659524</td>
          <td>0.215546</td>
          <td>25.270611</td>
          <td>0.333946</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.027891</td>
          <td>0.619318</td>
          <td>26.809292</td>
          <td>0.208659</td>
          <td>26.655020</td>
          <td>0.165225</td>
          <td>26.548156</td>
          <td>0.242050</td>
          <td>26.490351</td>
          <td>0.411804</td>
          <td>27.129746</td>
          <td>1.174251</td>
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
          <td>1.398944</td>
          <td>27.475514</td>
          <td>0.762694</td>
          <td>26.497975</td>
          <td>0.138265</td>
          <td>25.930345</td>
          <td>0.074150</td>
          <td>25.218085</td>
          <td>0.064430</td>
          <td>24.825584</td>
          <td>0.087052</td>
          <td>24.006474</td>
          <td>0.095064</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.486867</td>
          <td>0.768760</td>
          <td>27.186102</td>
          <td>0.247284</td>
          <td>26.866714</td>
          <td>0.167704</td>
          <td>26.103780</td>
          <td>0.140166</td>
          <td>25.808111</td>
          <td>0.203338</td>
          <td>25.550588</td>
          <td>0.348635</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.101455</td>
          <td>0.618443</td>
          <td>29.927644</td>
          <td>1.617893</td>
          <td>28.720025</td>
          <td>0.752998</td>
          <td>25.893651</td>
          <td>0.127025</td>
          <td>25.047920</td>
          <td>0.114742</td>
          <td>24.200135</td>
          <td>0.122460</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>31.527472</td>
          <td>4.281622</td>
          <td>28.528159</td>
          <td>0.797298</td>
          <td>26.984791</td>
          <td>0.229527</td>
          <td>26.418131</td>
          <td>0.228982</td>
          <td>25.970180</td>
          <td>0.286725</td>
          <td>25.983043</td>
          <td>0.587685</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.028005</td>
          <td>0.258938</td>
          <td>26.039665</td>
          <td>0.092892</td>
          <td>25.839390</td>
          <td>0.068505</td>
          <td>25.750824</td>
          <td>0.103196</td>
          <td>25.433379</td>
          <td>0.147976</td>
          <td>25.100877</td>
          <td>0.242673</td>
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
          <td>0.389450</td>
          <td>26.455516</td>
          <td>0.382655</td>
          <td>26.378552</td>
          <td>0.133463</td>
          <td>25.383557</td>
          <td>0.049441</td>
          <td>25.103913</td>
          <td>0.063275</td>
          <td>24.884781</td>
          <td>0.099192</td>
          <td>24.759809</td>
          <td>0.197067</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.373685</td>
          <td>1.314853</td>
          <td>26.919888</td>
          <td>0.200753</td>
          <td>26.148605</td>
          <td>0.091374</td>
          <td>25.124070</td>
          <td>0.060313</td>
          <td>24.776830</td>
          <td>0.084771</td>
          <td>24.202692</td>
          <td>0.114783</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.750701</td>
          <td>0.470232</td>
          <td>26.540082</td>
          <td>0.149472</td>
          <td>26.360065</td>
          <td>0.113570</td>
          <td>26.325906</td>
          <td>0.177985</td>
          <td>26.153583</td>
          <td>0.282835</td>
          <td>25.698772</td>
          <td>0.408790</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.906202</td>
          <td>0.548586</td>
          <td>26.383129</td>
          <td>0.138260</td>
          <td>26.172946</td>
          <td>0.102967</td>
          <td>25.825335</td>
          <td>0.123836</td>
          <td>25.289110</td>
          <td>0.146019</td>
          <td>25.784902</td>
          <td>0.462719</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.131084</td>
          <td>1.162161</td>
          <td>26.840736</td>
          <td>0.191381</td>
          <td>26.398794</td>
          <td>0.116306</td>
          <td>26.435106</td>
          <td>0.193244</td>
          <td>25.745091</td>
          <td>0.199975</td>
          <td>25.489244</td>
          <td>0.344124</td>
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
